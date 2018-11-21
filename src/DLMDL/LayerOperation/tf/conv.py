from ..layer_operation import LayerOperation
import tensorflow as tf
import re
from initializer import get_initializer
from regularizer import get_regularizer

# WARNING: Only 2D convolution available
class op_tf_conv(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define convolution operation for input tensor
        outputs:
            output: convolution output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        kernel_size = self.get_attr('kernel_size', default=None)
        if kernel_size is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('kernel_size', self.name))
        num_output = self.get_attr('num_output', default=None)
        if num_output is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_output', self.name))

        # optional field
        padding = self.get_attr('padding', default='VALID')
        stride = self.get_attr('stride', default=1)
        bias_term = self.get_attr('bias_term', default=True)
        initializer = self.get_attr('initializer', default={'weight': {}, 'bias':{}}) # default will set later
        regularizer = self.get_attr('regularizer', default={}) # default will set later
        dilate = self.get_attr('dilate', default=None)
        scope = self.get_attr('scope', default=self.name)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # get shape array
        stride_shape = [stride, stride]
        weight_shape = [kernel_size[0], kernel_size[1], indim[3], num_output]
        dilate_shape = [dilate, dilate] if dilate is not None else None
        bias_shape = [num_output]

        with tf.variable_scope(self.name):
            # get weight for convolution
            with tf.variable_scope(scope):
                weight_init = get_initializer(initializer.get('weight'), is_bias=False)
                weight_reg = get_regularizer(regularizer, is_bias=False)
                weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                          initializer=weight_init, regularizer=weight_reg,
                                          trainable=True)
                #tf.add_to_collection(scope, weights)

                if bias_term:
                    bias_init = get_initializer(initializer.get('bias'), is_bias=True)
                    bias_reg = get_regularizer(regularizer, is_bias=True)
                    biases = tf.get_variable('biases', shape=bias_shape, dtype=tf.float32,
                                             initializer=bias_init, regularizer=bias_reg,
                                             trainable=True)
                    #tf.add_to_collection(scope, biases)

        # construct API
        def apiConstructor():


            conv = tf.nn.convolution(input_, weights, padding,
                                     strides=stride_shape, dilation_rate=dilate_shape, data_format='NHWC')

            # if bias_term is True, add bias term to convolution output
            if bias_term:
                conv = tf.nn.bias_add(conv, biases, data_format='NHWC')

            # get output dimension
            outdim = list(conv.get_shape()[i].value for i in xrange(len(conv.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', conv)

            # set tf summary
            tf.summary.histogram(self.name, conv)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()