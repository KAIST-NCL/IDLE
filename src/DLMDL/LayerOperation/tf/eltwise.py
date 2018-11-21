from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_eltwise(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define element-wise operation for input tensors
        ourputs:
            output: eltwise output
        """
        # get input
        input_ = self.get_input('input') #WARNING: number of inputs must be 2
        indim = self.get_dimension('input')

        # get attr
        # required field
        op = self.get_attr('operation', default=None)
        if op is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('op', self.name))
        # optional field
        scale = float(self.get_attr('scale', default=1.0))
        scope = self.get_attr('scope', default=None)
        # TODO: tmp
        if scope is None:
            scope = self.name

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            if op == 'MUL':
                eltwise = tf.multifly(input_[0], input_[1])
            elif op == 'SUM':
                eltwise = scale * tf.add(input_[0], input_[1])
            elif op == 'MAX':
                eltwise = tf.maximum(input_[0], input_[1])

            # get output dimension
            outdim = list(eltwise.get_shape()[i].value for i in xrange(len(eltwise.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', eltwise)

            # set tf summary
            # WARNING: TMP!!!!!
            if len(outdim) == 0:
                if scope is not None:
                    tf.summary.scalar(self.name, eltwise, collections=[scope])
                else:
                    tf.summary.scalar(self.name, eltwise)
            else:
                tf.summary.histogram(self.name, eltwise)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()