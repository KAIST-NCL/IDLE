# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_c_max_pool(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "ksize"}, \
    {"source": "layer", "mandatory": "both", "name": "stride"}, \
    {"default": "SAME", "source": "layer", "mandatory": "both", "name": "padding"}, \
    {"default": "NHWC", "source": "layer", "mandatory": "tf", "name": "data_format"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"}, \
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""


    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, ksize, stride, padding, activation):
            ksize = [1, ksize, ksize, 1]
            data_format = self.get_attr('data_format', self.data_format)
            if data_format == 'NCHW':
                strides = [1, 1, stride, stride]
            else:
                strides = [1, stride, stride, 1]
            maxpool_ = tf.nn.max_pool(input_, ksize, strides, padding, data_format=data_format, name='pool')

            for act in activation:
                if act == 'relu':
                    maxpool_ = tf.nn.relu(maxpool_)
            return maxpool_

        ksize = self.get_attr('ksize')
        stride = self.get_attr('stride')
        padding = self.get_attr('padding')
        input_ = self.get_input('input')
        activation = self.get_attr('activation',self.activation)

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            if learning_option.get("parallel") != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    maxpool_ = apiConstructor(input_, ksize, stride, padding, activation)
            else:
                maxpool_ = apiConstructor(input_, ksize, stride, padding, activation)
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
                maxpool_ = tf.contrib.layers.batch_norm(maxpool_,
                                                     decay=decay,
                                                     center=center,
                                                     scale=scale,
                                                     epsilon=epsilon,
                                                     is_training=is_training)
            tf.summary.histogram('activations', maxpool_)
            outdim = list(maxpool_.get_shape()[i].value for i in xrange(len(maxpool_.get_shape())))
            self.set_dimension('output', outdim)
            self.set_output('output', maxpool_)