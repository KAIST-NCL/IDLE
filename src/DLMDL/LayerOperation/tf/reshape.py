from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_reshape(LayerOperation):
    _attributes = """[]"""  # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define reshape operation for input tensor
        outputs:
            output: reshape output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        shape = self.get_attr('shape', default=None)
        if shape is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('shape', self.name))

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            reshape = tf.reshape(input_, shape)

            # get output dimension
            outdim = list(reshape.get_shape()[i].value for i in xrange(len(reshape.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', reshape)

            # set tf summary
            tf.summary.histogram(self.name, reshape)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()