# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_gradientdescentoptimizer(LayerOperation):

    _attributes = """[{"default": 0.001, "source": "opt", "mandatory": "both", "name": "learning_rate"}, {"default": false, "mandatory": "tf", "name": "use_locking"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConsturctor(self, input_, learning_rate, use_locking):
            gradopt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=use_locking)
            gradopt_ = gradopt.minimize(input_, colocate_gradients_with_ops=True, global_step=global_step)
            return gradopt_

        learning_rate = learning_option.get("learning_rate", self.learning_rate)
        use_locking = learning_option.get("use_locking", self.use_locking)
        input_ = self.get_input('loss')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            global_step = tf.train.get_or_create_global_step()
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    gradopt_ = apiConsturctor(input_, learning_rate, use_locking)
            else:
                gradopt_ = apiConsturctor(input_, learning_rate, use_locking)
            self.set_output('output', gradopt_)
            self.set_output('global_step', global_step)
