from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_rnn(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define rnn cell for input tensor
        outputs:
            output: rnn cell
        """
        # get input
        # TODO: TO BE DEPRECATED
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        #TODO: input(data tensor) will input to static_rnn layer.
        learning_option['static_rnn_input'] = input_

        # get attr
        # optional field
        activation = self.get_attr('activation', default=None)
        scope = self.get_attr('scope', default=None)
        # TODO: tmp
        if scope is None:
            scope = self.name

        hidden_size = learning_option.get('hidden_size') # num. of cells
        is_train = learning_option.get('is_train')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            #TODO: RNN cudnn cell(CUDNN implmentation of RNN layer)

            """
            TODO: add conditional branch later. TF v1.1 cannot add branch since name attribute not exist in BasicRNNCell API.
            def f1(): return tf.contrib.rnn.BasicRNNCell(hidden_size, activation=activation) # for training procedure
            def f2(): return tf.contrib.rnn.BasicRNNCell(hidden_size, activation=activation,
                                               reuse=True) # for test procedure
            rnn_cell = tf.cond(is_train, f1, f2, name=self.name)
            """

            rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_size, activation=activation) # for training procedure

            # set output
            self.set_output('output', rnn_cell)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()

