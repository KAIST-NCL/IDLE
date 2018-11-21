from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_dropout(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define dropout operation for input tensor
        outputs:
            output: dropout output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        dropout_ratio = float(self.get_attr('dropout_ratio', default=0.0))
        if dropout_ratio is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('dropout_ratio', self.name))

        is_train = learning_option.get('is_train')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        def apiConstructor():
            # during training procedure dropout on, during testing procedure dropout off
            dropout = tf.cond(is_train, lambda: tf.nn.dropout(input_, 1.0-dropout_ratio),
                              lambda: tf.nn.dropout(input_, 1.0))

            # get output dimension
            outdim = list(dropout.get_shape()[i].value for i in xrange(len(dropout.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', dropout)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()