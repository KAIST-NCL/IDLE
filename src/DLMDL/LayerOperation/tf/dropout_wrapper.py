from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_dropout_wrapper(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define dropout wrapping of lstm/rnn/gru cells for input tensor
        outputs:
            output: dropout wrapper
        """
        # get input
        # TODO: to be change 'input' to 'cell
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        input_dropout_ratio = self.get_attr('input_dropout_ratio', default=None)
        if input_dropout_ratio is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('input_dropout_ratio', self.name))

        # optional field
        output_dropout_ratio = float(self.get_attr('output_dropout_ratio', default=0.0))
        state_dropout_ratio = float(self.get_attr('state_dropout_ratio', default=0.0))

        is_train = learning_option.get('is_train')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():

            """
            TODO: add conditional branch later. TF v1.1 cannot add branch since name attribute not exist in DropoutWrapper API.
            def f1(): return tf.contrib.rnn.DropoutWrapper(input_, input_keep_prob= 1.0-input_dropout_ratio
                                                    ,output_keep_prob=1.0-output_dropout_ratio
                                                    ,state_keep_prob=1.0-state_dropout_ratio) # for training procedure
            def f2(): return input_ # for testing procedure
            dropout_wrapper = tf.cond(is_train, f1, f2)

            """
            dropout_wrapper = tf.contrib.rnn.DropoutWrapper(input_, input_keep_prob= 1.0-input_dropout_ratio
                                                            ,output_keep_prob=1.0-output_dropout_ratio
                                                            ,state_keep_prob=1.0-state_dropout_ratio) # for training procedure

            # set output
            self.set_output('output', dropout_wrapper)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()

