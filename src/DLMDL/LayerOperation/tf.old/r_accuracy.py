# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_r_accuracy(LayerOperation):

    _attributes = """[]"""


    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(logits, labels, topk):
            predictions = tf.nn.softmax(logits)
            correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            return accuracy

        logits = self.get_input('logits')
        labels = self.get_input('labels')
        topk = self.get_attr('topk', 1)#TODO: topk implement-twkim
        
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            accuracy = apiConstructor(logits, labels, topk)
           
            tf.summary.scalar('accuracy', accuracy)
            self.set_output('output', accuracy)
