from ..layer_operation import LayerOperation
import tensorflow as tf
import re
from initializer import get_initializer
from regularizer import get_regularizer

class op_tf_prelu(LayerOperation):
    _attributes = """[]"""  # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define parame rectified-linear unit(ReLU) operation for input tensor
        It follows: f(x) = alpha * x for x < 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.
        outputs:
            output: PReLU output
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        initializer = self.get_attr('initializer', default={'weight': {}, 'bias': {}})  # default will set later
        regularizer = self.get_attr('regularizer', default={})  # default will set later
        ch_shared = self.get_attr('channel_shared', default=False)
        scope = self.get_attr('scope', default='default')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            # get weight for prelu
            alpha_init = get_initializer(initializer.get('weight'), is_bias=False)
            alpha_reg = get_regularizer(regularizer, scope, is_bias=False)

            #WARNINIG: constraint of weight is always None
            prelu = tf.keras.layers.PReLU(input_, alpha_initializer=alpha_init,
                                          alpha_regularizer=alpha_reg,
                                          alpha_constraint=None,
                                          shared_axes=ch_shared)

            # get output dimension
            outdim = indim

            # set output
            self.set_dimension('output', outdim)
            self.set_output('output', prelu)

            # set tf summary
            tf.summary.histogram(self.name, prelu)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()