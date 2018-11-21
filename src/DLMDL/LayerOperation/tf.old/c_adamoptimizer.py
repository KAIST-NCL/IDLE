# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_c_adamoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.9, "source": "opt", "mandatory": "tf", "name": "beta1"},\
     {"default": 0.999, "source": "opt", "mandatory": "tf", "name": "beta2"}, \
     {"default": 1e-08, "source": "opt", "mandatory": "tf", "name": "epsilon"}]"""



    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, learning_rate, beta1, beta2, epsilon):
            adamopt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
            adamopt_ = adamopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            return adamopt_

        learning_rate = learning_option.get("learning_rate")
        beta1 = learning_option.get("beta1", self.beta1)
        beta2 = learning_option.get("beta2", self.beta2)
        epsilon = learning_option.get("epsilon", self.epsilon)
        input_ = self.get_input('loss')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            global_step = tf.train.get_or_create_global_step()
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    adamopt_ = apiConstructor(input_, learning_rate, beta1, beta2, epsilon)
            else:
                adamopt_ = apiConstructor(input_, learning_rate, beta1, beta2, epsilon)
            self.set_output('output', adamopt_)
            self.set_output('global_step', global_step)