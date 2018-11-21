from ..layer_operation import LayerOperation
import tensorflow as tf
import re
from initializer import get_initializer
from regularizer import get_regularizer


class op_tf_fc(LayerOperation):
    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define fully-connected(FC) operation for input tensor.
        outputs:
            output: FC output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        num_output = self.get_attr('num_output', default=None)
        if num_output is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_output', self.name))

        # optional field
        bias_term = self.get_attr('bias_term', default=True)
        initializer = self.get_attr('initializer', default={'weight': {}, 'bias':{}})  # default will set later
        regularizer = self.get_attr('regularizer', default={})  # default will set later
        scope = self.get_attr('scope', default=None)
        #TODO: tmp
        if scope is None:
            scope = self.name

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            if learning_option.get('num_steps') == None: # DNN/CNN case
                # if this layer is first fc, flatten input
                if len(indim) == 2:
                    weight_shape = [indim[1], num_output]
                    flatten = input_
                else:
                    flatten = tf.reshape(input_, [-1, indim[1]*indim[2]*indim[3]])
                    weight_shape = [flatten.get_shape()[1].value, num_output]
                with tf.variable_scope(scope):
                    # get weight for fc
                    weight_init = get_initializer(initializer.get('weight'), is_bias=False)
                    weight_reg = get_regularizer(regularizer, is_bias=False)
                    weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                          initializer=weight_init, regularizer=weight_reg,
                                          trainable=True)
                    tf.add_to_collection(scope, weights)

                fc =  tf.matmul(flatten, weights)

                # if bias_term is True, add bias term to fc output
                if bias_term:
                    with tf.variable_scope(scope):
                        bias_shape = [num_output]
                        bias_init = get_initializer(initializer.get('bias'), is_bias=True)
                        bias_reg = get_regularizer(regularizer, is_bias=True)
                        biases = tf.get_variable('biases', shape=bias_shape, dtype=tf.float32,
                                                 initializer=bias_init, regularizer=bias_reg,
                                                 trainable=True)
                        tf.add_to_collection(scope, biases)

                    fc = tf.nn.bias_add(fc, biases, data_format='NHWC')

            #WARNING: in recurrent neural network, there is only one fully-connected layer
            else: # RNN/LSTM/GRU case
                hidden_size = learning_option.get('hidden_size')
                weight_shape = [hidden_size, num_output]

                # get weight for fc
                with tf.variable_scope(scope):
                    weight_init = get_initializer(initializer.get('weight'), is_bias=False)
                    weight_reg = get_regularizer(regularizer, is_bias=False)
                    weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                              initializer=weight_init, regularizer=weight_reg,
                                              trainable=True)
                    tf.add_to_collection(scope, weights)

                if learning_option.get('is_image'): # MNIST rnn
                    fc = tf.matmul(input_[-1], weights)
                else:
                    reshape_input_ = tf.reshape(tf.stack(axis=1, values=input_), [-1, hidden_size])
                    fc = tf.matmul(reshape_input_, weights)

                # if bias_term is True, add bias term to fc output
                if bias_term:
                    with tf.variable_scope(scope):
                        bias_shape = [num_output]
                        bias_init = get_initializer(initializer.get('bias'), is_bias=True)
                        bias_reg = get_regularizer(regularizer, is_bias=True)
                        biases = tf.get_variable('biases', shape=bias_shape, dtype=tf.float32,
                                                 initializer=bias_init, regularizer=bias_reg,
                                                 trainable=True)
                        tf.add_to_collection(scope, biases)
                    fc = tf.nn.bias_add(fc, biases, data_format='NHWC')

            # get output dimension
            outdim = list(fc.get_shape()[i].value for i in xrange(len(fc.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', fc)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()
