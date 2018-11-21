# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_r_loss(LayerOperation):

    _attributes = """[]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        logits = self.get_input('logits')
        labels = self.get_input('labels')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')
        softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        #print('softmax_cross_entropy_with_logits = {0}'.format(softmax_cross_entropy_with_logits))
        
        loss = tf.reduce_mean(softmax_cross_entropy_with_logits)
        #print('loss = {0}'.format(loss))
        
        tf.summary.scalar('loss', loss)
        self.set_output('output', loss)
