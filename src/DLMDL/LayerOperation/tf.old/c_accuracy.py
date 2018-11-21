# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_c_accuracy(LayerOperation):

    _attributes = """[\
    {"default": 1, "source": "topk", "mandatory": "both", "name": "topk"}]"""


    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(logits, labels, topk):
            predictions = tf.argmax(logits, axis=1)
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
            return accuracy

        logits = self.get_input('logits')
        labels = tf.cast(self.get_input('labels'), tf.int64)
        topk = self.get_attr('topk', self.topk)

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    accuracy = apiConstructor(logits, labels, topk)
            else:
                accuracy = apiConstructor(logits, labels, topk)

            tf.summary.scalar('accuracy', accuracy)
            self.set_output('output', accuracy)
