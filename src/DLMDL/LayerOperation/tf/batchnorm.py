from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_batchnorm(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define batch normalization operation for input tensor
        outputs:
            output: batch normalization output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        eps = float(self.get_attr('epsilon', default=10**-5))
        moving_avg = float(self.get_attr('moving_average', default=0.999))
        scope = self.get_attr('scope', default='default')

        is_train = learning_option.get('is_train')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        def apiConstructor():
            batchnorm = tf.layers.batch_normalization(input_, training=is_train)

            # get output dimension
            outdim = list(batchnorm.get_shape()[i].value for i in xrange(len(batchnorm.get_shape())))

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', batchnorm)

            # set tf summary
            tf.summary.histogram(self.name, batchnorm)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) == "MP" or learning_option.get("parallel", None) is None:
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            elif learning_option.get("parallel", None) == "DP_mb":
                with tf.device('/job:worker/task:{0}/mb:0'.format(device)):
                    apiConstructor()
            else:
                apiConstructor()