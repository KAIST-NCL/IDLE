# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_c_eltwise(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "eltwise_op"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"}, \
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, eltwise_op, activation):
            if eltwise_op == 'SUM':
                eltwise_ = tf.add(input_[0], input_[1])
            else:
                if eltwise_op == "PROD":
                    eltwise_ = tf.multiply(input_[0], input_[1])
                if eltwise_op == "MAX":
                    eltwise_ = tf.maximum(input_[0], input_[1])
                else:
                    # TODO: error handling?
                    pass

            if len(activation) != 0:
                for act in activation:
                    if act == 'relu':
                        eltwise_ = tf.nn.relu(eltwise_)
            return eltwise_

        input_ = self.get_input('input')
        eltwise_op = self.get_attr('eltwise_op')
        activation = self.get_attr('activation',self.activation)

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            if learning_option.get("parallel", None) != "DP":
                #with tf.device('/job:worker/task:0'):
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    eltwise_ = apiConstructor(input_, eltwise_op, activation)
            else:
                eltwise_ = apiConstructor(input_, eltwise_op, activation)
            if 'batchnorm' in activation:
                is_training = self.get_attr('use_global_stats', self.use_global_stats)
                decay = self.get_attr('moving_average_fraction', self.moving_average_fraction)
                epsilon = self.get_attr('epsilon', self.epsilon)
                scale = False
                center = False
                if self.get_attr('is_scale', self.is_scale) == True:
                    scale = True
                    center = self.get_attr('bias_term', self.bias_term)
                # batch normalization and sacling activation
                eltwise_ = tf.contrib.layers.batch_norm(eltwise_,
                                                     decay=decay,
                                                     center=center,
                                                     scale=scale,
                                                     epsilon=epsilon,
                                                     is_training=is_training)
            self.set_output('output', eltwise_)
