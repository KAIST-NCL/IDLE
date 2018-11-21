# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_c_fc(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "output_shape"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"}, \
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConsturctor(input_, weights, biases, activation):
            fc_ = tf.add(tf.matmul(input_, weights), biases)
            # fc_ = tf.nn.relu_layer(input_, weights, biases)
            if len(activation) != 0:
                for act in activation:
                    if act == 'relu':
                        fc_ = tf.nn.relu(fc_)
            return fc_

        output_shape = self.get_attr('output_shape') # mandatory
        if output_shape != None:
            learning_option['classes'] =  output_shape
            
        activation = self.get_attr('activation',self.activation)
        input_ = self.get_input('input')
        mean = self.get_attr('mean', 0) # not added
        stddev = self.get_attr('stddev', 1e-1) # not added
        value = self.get_attr('value', 0.0) # not added
        indim = self.get_dimension('input')

        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        with tf.name_scope(self.name) as scope:
            if len(indim) == 2:
                nIn = indim[1]
            else:
                input_ = tf.reshape(input_, [-1,indim[1]*indim[2]*indim[3]])
                nIn = input_.get_shape()[1].value
            weights = tf.Variable(
                tf.truncated_normal([nIn, output_shape], dtype=tf.float32, mean=mean, stddev=stddev, name='weight'))
            biases = tf.Variable(tf.constant(value, dtype=tf.float32, shape=[output_shape]), name='biase')
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    fc_ = apiConsturctor(input_, weights, biases, activation)
            else:
                fc_ = apiConsturctor(input_, weights, biases, activation)
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
                fc_ = tf.contrib.layers.batch_norm(fc_,
                                                     decay=decay,
                                                     center=center,
                                                     scale=scale,
                                                     epsilon=epsilon,
                                                     is_training=is_training)

            tf.summary.histogram('activations', fc_)
            outdim = list(fc_.get_shape()[i].value for i in xrange(len(fc_.get_shape())))
            self.set_dimension('output', outdim)
            self.set_output('output', fc_)
 