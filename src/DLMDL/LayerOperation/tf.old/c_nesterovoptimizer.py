# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_nesterovoptimizer(LayerOperation):

    # in tensorflow, nesterovoptimizer is implemented using tf.datasets.MomentumOptimizer(....,use_nestrov=True)
    _attributes = """[\
    {"default": 0.9, "source": "opt", "mandatory": "both", "name": "momentum"}]"""


    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, learning_rate, momentum, use_nesterov):
            nesterovopt = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=use_nesterov)
            nesterovopt_ = nesterovopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            return nesterovopt_

        learning_rate = learning_option.get("learning_rate")
        momentum = learning_option.get("momentum", self.momentum)
        use_nesterov = True # use nesterov optimizer using nesterovoptimizer
        input_ = self.get_input('loss')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            global_step = tf.train.get_or_create_global_step()
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    nesterovopt_ = apiConstructor(input_, learning_rate, momentum, use_nesterov)
            else:
                nesterovopt_ = apiConstructor(input_, learning_rate, momentum, use_nesterov)
            self.set_output('output', nesterovopt_)
            self.set_output('global_step', global_step)