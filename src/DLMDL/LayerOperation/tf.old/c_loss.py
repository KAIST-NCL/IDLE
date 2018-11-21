# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_loss(LayerOperation):

    _attributes = """[]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(logits, labels):
            softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                               labels=labels)
            loss = tf.reduce_mean(softmax_cross_entropy_with_logits)
            return loss

        logits = self.get_input('logits')
        labels = tf.cast(self.get_input('labels'), tf.int64)

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    loss = apiConstructor(logits, labels)
            else:
                loss = apiConstructor(logits, labels)
            tf.summary.scalar('loss', loss)
            self.set_output('output', loss)
