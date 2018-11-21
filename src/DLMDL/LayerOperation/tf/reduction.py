from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_reduction(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define reduction operation for input tensors
        ourputs:
            output: reduction output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        op = self.get_attr('operation', default=None)
        if op is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('op', self.name))

        # optional field
        axis = self.get_attr('axis', default=None)
        scale = float(self.get_attr('scale', default=1.0))
        keep_dim = self.get_attr('keep_dim', default=False)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            if op == 'SUM':
                reduction = scale * tf.reduce_sum(input_, axis=axis,
                                                  keepdims=keep_dim)
            elif op == 'ASUM':
                reduction = scale * tf.reduce_sum(tf.abs(input_), axis=axis,
                                                  keepdims=keep_dim)
            elif op == 'SUMSQ':
                reduction = scale * tf.reduce_sum(tf.square(input_), axis=axis,
                                                  keepdims=keep_dim)
            elif op == 'MEAN':
                reduction = scale * tf.reduce_mean(input_, axis=axis,
                                                   keepdims=keep_dim)

            # get output dimension
            outdim = list(reduction.get_shape()[i].value for i in xrange(len(reduction.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', reduction)

            # set tf summary
            tf.summary.histogram(self.name, reduction)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()