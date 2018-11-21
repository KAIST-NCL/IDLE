# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_c_conv(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "filter"}, \
    {"source": "layer", "mandatory": "both", "name": "stride"}, \
    {"default": "SAME", "source": "layer", "mandatory": "both", "name": "padding"}, \
    {"default": "NHWC", "source": "layer", "mandatory": "tf", "name": "data_format"}, \
    {"default": true, "source": "layer", "mandatory": "tf", "name": "use_cudnn_on_gpu"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"}, \
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(input_, kernel, biases, stride, padding, activation, data_format, use_cudnn_on_gpu):
            if data_format == 'NCHW':
                strides = [1, 1, stride, stride]
            else:
                strides = [1, stride, stride, 1]

            conv = tf.nn.conv2d(input_, kernel, strides, padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format)
            conv_ = tf.nn.bias_add(conv, biases, data_format=data_format)
            if len(activation) != 0:
                for act in activation:
                    if act == 'relu':
                        conv_ = tf.nn.relu(conv_)
            return conv_
        
        filter = self.get_attr('filter')
        stride = self.get_attr('stride')
        padding = self.get_attr('padding')
        activation = self.get_attr('activation',self.activation)
        data_format = self.get_attr('data_format', self.data_format)
        use_cudnn_on_gpu = self.get_attr('use_cudnn_on_gpu', self.use_cudnn_on_gpu)
        mean = self.get_attr('mean', 0) # not added
        stddev = self.get_attr('stddev', 1e-2) # not added
        value = self.get_attr('value', 0.0) # not added
        indim = self.get_dimension('input')
        input_ = self.get_input('input')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            kernel = tf.Variable(
                tf.truncated_normal([filter[0], filter[1], filter[2], filter[3]], dtype=tf.float32, mean=mean,
                                    stddev=stddev, name='weights'))
            biases = tf.Variable(tf.constant(value, shape=[filter[3]], dtype=tf.float32), name='biases')
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    conv_ = apiConstructor(input_, kernel, biases, stride, padding, activation, data_format, use_cudnn_on_gpu)

                tf.summary.histogram('activations', conv_)
                outdim = list(conv_.get_shape()[i].value for i in xrange(len(conv_.get_shape())))
                self.set_dimension('output', outdim)
                self.set_output('output', conv_)
            else:
                conv_ = apiConstructor(input_, kernel, biases, stride, padding, activation, data_format, use_cudnn_on_gpu)

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
                conv_ = tf.contrib.layers.batch_norm(conv_,
                                                     decay=decay,
                                                     center=center,
                                                     scale=scale,
                                                     epsilon=epsilon,
                                                     is_training=is_training)

            tf.summary.histogram('activations', conv_)
            outdim = list(conv_.get_shape()[i].value for i in xrange(len(conv_.get_shape())))
            self.set_dimension('output', outdim)
            self.set_output('output', conv_)