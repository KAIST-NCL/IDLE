# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_adadeltaoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.95, "source": "tf", "mandatory": "both", "name": "rho"},\
    {"default": 1e-08, "source": "opt", "mandatory": "both", "name": "epsilon"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, learning_rate, rho, epsilon):
            adadeltaopt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho, epsilon=epsilon)
            adadeltaopt_ = adadeltaopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            return adadeltaopt_

        learning_rate = learning_option.get("learning_rate")
        rho = learning_option.get("rho", self.rho)
        epsilon = learning_option.get("epsilon", self.epsilon)
        input_ = self.get_input('loss')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            global_step = tf.train.get_or_create_global_step()
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    adadeltaopt_ = apiConstructor(input_, learning_rate, rho, epsilon)
            else:
                adadeltaopt_ = apiConstructor(input_, learning_rate, rho, epsilon)
            self.set_output('output', adadeltaopt_)
            self.set_output('global_step', global_step)