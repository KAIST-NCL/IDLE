# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_r_sequenceloss(LayerOperation):

    _attributes = """[{"default": 1.0, "source":"layer", "mandatory": "tf", "name":"weights"}, \
     {"default": false, "source":"layer", "mandatory":"tf", "name":"average_across_timesteps"}, \
     {"default": true, "source":"layer", "mandatory":"tf", "name":"average_across_batch"}, \
     {"default": false, "source":"layer", "mandatory":"tf", "name":"softmax_loss_function"} \
    ]"""


    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(logits, targets, weights, avg_across_ts, avg_across_batch, softmax_loss_ft):
            logits_ = tf.reshape(logits, [learning_option.get('batch_size'), learning_option.get('num_steps'),
                                          learning_option.get('num_class')])
            loss = tf.contrib.seq2seq.sequence_loss(logits_,
                                                    targets,
                                                    tf.ones([learning_option.get('batch_size'), learning_option.get('num_steps')], dtype=tf.float32),
                                                    average_across_timesteps=False,
                                                    average_across_batch=True)

            loss = tf.reduce_sum(loss)
            return loss

        logits = self.get_input('logits')
        targets = self.get_input('targets')

        weights = self.get_attr('weights')
        avg_across_ts = self.get_attr('average_across_timesteps')
        avg_across_batch = self.get_attr('average_across_batch')
        softmax_loss_ft =  self.get_attr('softmax_loss_function')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            loss = apiConstructor(logits, targets, weights, avg_across_ts, avg_across_batch, softmax_loss_ft)
            tf.summary.scalar('sequence_loss', loss)
            self.set_output('output', loss)
