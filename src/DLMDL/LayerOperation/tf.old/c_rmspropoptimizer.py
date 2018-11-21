# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_c_rmspropoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.9, "source": "opt", "mandatory": "both", "name": "decay"},\
    {"default": 0.9, "source": "opt", "mandatory": "both", "name": "momentum"},\
    {"default": 1e-10, "source": "both", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "mandatory": "tf", "name": "centered"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, learning_rate, decay, momentum, epsilon, centered):
            rmsopt = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon,
                                               centered=centered)
            rmsopt_ = rmsopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            return rmsopt_

        learning_rate = learning_option.get("learning_rate")
        decay = learning_option.get("decay", self.decay)
        momentum = learning_option.get("momentum", self.momentum)
        epsilon = learning_option.get("epsilon", self.epsilon)
        centered = learning_option.get("centered", self.centered)
        input_ = self.get_input('loss')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            global_step = tf.train.get_or_create_global_step()
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    rmsopt_ = apiConstructor(input_, learning_rate, decay, momentum, epsilon, centered)

            else:
                rmsopt_ = apiConstructor(input_, learning_rate, decay, momentum, epsilon, centered)
            self.set_output('output', rmsopt_)
            self.set_output('global_step', global_step)