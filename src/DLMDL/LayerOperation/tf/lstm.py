from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_lstm(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define lstm cell for input tensor
        outputs:
            output: lstm cell
        """
        # get input
        # TODO: TO BE DEPRECATED
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        #TODO: input(data tensor) will input to static_rnn layer.
        learning_option['static_rnn_input'] = input_

        # get attr
        # optional field
        forget_bias = float(self.get_attr('forget_bias', default=1.0))
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

            #TODO: LSTM cudnn cell(CUDNN implmentation of LSTM layer)
            """
            TODO: add conditional branch later. TF v1.1 cannot add branch since name attribute not exist in BasicLSTMCell API.
            def f1(): return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias,
                                                      state_is_tuple=True, activation=activation) # for training procedure
            def f2(): return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias,
                                               state_is_tuple=True, activation=activation,
                                               reuse=True) # for test procedure
            lstm_cell = tf.cond(is_train, f1, f2, name=self.name)
            """

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias, activation=activation,
                                                     state_is_tuple=True,  reuse=tf.get_variable_scope().reuse) # for training procedure

            # set output
            self.set_output('output', lstm_cell)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()

