# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_concat(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "concat_axis"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, axis):
            concat_ = tf.concat(input_, axis)
            return concat_
        input_ = self.get_input('input')
        axis = self.get_attr('concat_axis')
        axis = 3 if axis == 'ch' else  0 # TODO: error check (e.g. "wdefw")??
        #in tensorflow, ch means 3, num means 0 (NCHW structure)

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    concat_ = apiConstructor(input_, axis)
            else:
                concat_ = apiConstructor(input_, axis)
            self.set_output('output', concat_)