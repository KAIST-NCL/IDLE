# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_dropout(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "dropout_ratio"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, keep_prob):
            drop = tf.nn.dropout(input_, keep_prob)
            return drop

        input_ = self.get_input('input')
        keep_prob = 1 - self.get_attr('dropout_ratio', self.dropout_ratio)

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            #if learning_option.get("parallel", None) != "DP":
            #    with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
            #        drop = apiConstructor(input_, keep_prob)
            #else:
            drop = apiConstructor(input_, keep_prob)
            #tf.summary.scalar('dropout', drop)
            outdim = list(drop.get_shape()[i].value for i in xrange(len(drop.get_shape())))
            self.set_dimension('output', outdim)
            self.set_output('output', drop)
