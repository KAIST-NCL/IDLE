# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_adagradoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.1, "source": "opt", "mandatory": "tf", "name": "initial_accumulator_value"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, learning_rate, initial_accumulator_value):
            adagradopt = tf.train.AdagradOptimizer(learning_rate,
                                                   initial_accumulator_value=initial_accumulator_value)
            adagradopt_ = adagradopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            return adagradopt_

        learning_rate = learning_option.get("learning_rate")
        initial_accumulator_value = learning_option.get("initial_accumulator_value", self.initial_accumulator_value)
        input_ = self.get_input('loss')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            global_step = tf.train.get_or_create_global_step()
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    adagradopt_ = apiConstructor(input_, learning_rate, initial_accumulator_value)
            else:
                adagradopt_ = apiConstructor(input_, learning_rate, initial_accumulator_value)
            self.set_output('output', adagradopt_)
            self.set_output('global_step', global_step)

