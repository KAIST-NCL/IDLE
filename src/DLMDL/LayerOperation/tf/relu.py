from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_relu(LayerOperation):
    _attributes = """[]"""  # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define rectified-linear unit(ReLU) operation for input tensor
        outputs:
            output: ReLU output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        slope = float(self.get_attr('slope', default=0.0))

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            #TODO: tf.nn.leaky_relu is not supported in version 1.1.0. update later.
            relu = tf.nn.relu(input_)

            # get output dimension
            outdim = indim

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', relu)

            # set tf summary
            tf.summary.histogram(self.name, relu)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()