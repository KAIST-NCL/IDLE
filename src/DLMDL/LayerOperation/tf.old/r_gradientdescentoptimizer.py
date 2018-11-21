# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_r_gradientdescentoptimizer(LayerOperation):

    _attributes = """[{"default": 1, "mandatory": "tf", "name": "clip_norm"}, \
     {"default": false, "mandatory": "tf", "name": "use_locking"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConsturctor(input_, learning_rate, clip_norm, use_locking, global_step):
            if clip_norm == None or clip_norm == 0:
                gradopt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=use_locking)
                gradopt_ = gradopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            else:
                train_vars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(input_, train_vars), clip_norm)
                gradopt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=use_locking)
                gradopt_ = gradopt.apply_gradients(zip(grads, train_vars),
                                                   global_step=global_step)
            return gradopt_

        learning_rate = learning_option.get("learning_rate")
        use_locking = self.get_attr('use_locking')
        input_ = self.get_input('loss')

        clip_norm = self.get_attr('clip_norm')

        with tf.name_scope(self.name) as scope:
            #global_step = tf.Variable(0, trainable=False, dtype=tf.int32)   #TODO
            global_step = tf.train.get_or_create_global_step()
            gradopt_ = apiConsturctor(input_, learning_rate, clip_norm, use_locking, global_step)

            self.set_output('output', gradopt_)
            self.set_output('global_step', global_step)
