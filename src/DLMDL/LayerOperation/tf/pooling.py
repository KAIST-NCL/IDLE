from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_pooling(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define pooling(max/average pooling) operation for input tensor
        outputs:
            output: pooling output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        pool_type = self.get_attr('pool_type', default=None)
        if pool_type  is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('type', self.name))
        kernel_size = self.get_attr('kernel_size', default=None)
        if kernel_size is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('kernel_size', self.name))

        # optional field
        padding = self.get_attr('padding', default='VALID')
        stride = self.get_attr('stride', default=1)
        dilate = self.get_attr('dilate', default=None)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            # get shape array
            stride_shape = [stride, stride]
            dilate_shape = [dilate, dilate] if dilate is not None else None

            pool = tf.nn.pool(input_, kernel_size, pool_type, padding, dilation_rate=dilate_shape,
                              strides=stride_shape, data_format='NHWC')
            # get output dimension
            outdim = list(pool.get_shape()[i].value for i in xrange(len(pool.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', pool)

            # set tf summary
            tf.summary.histogram(self.name, pool)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()


