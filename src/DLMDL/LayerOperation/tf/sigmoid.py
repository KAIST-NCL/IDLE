from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_sigmoid(LayerOperation):
    _attributes = """[]"""  # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define sigmoid operation for input tensor
        Computes sigmoid of x element-wise. Specifically, y = 1 / (1 + exp(-x)).
        outputs:
            output: sigmoid output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            sigmoid = tf.sigmoid(input_)

            # get output dimension
            outdim = indim

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', sigmoid)

            # set tf summary
            tf.summary.histogram(self.name, sigmoid)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()