from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_concat(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define concat operation for input tensors
        ourputs:
            output: concat output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        axis = self.get_attr('axis', default=None)
        if axis is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('axis', self.name))

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            concat = tf.concat(input_, axis)

            # get output dimension
            outdim = list(concat.get_shape()[i].value for i in xrange(len(concat.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', concat)

            # set tf summary
            tf.summary.histogram(self.name, concat)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()