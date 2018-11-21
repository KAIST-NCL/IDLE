# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_r_fc(LayerOperation):

    _attributes = """[]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):

        size = learning_option['hidden_size']
        num_class = learning_option.get('num_class')

        input_ = self.get_input('input')
        
        if learning_option.get('file_format') == "mnist": # tmp- twkim
            #TODO: initializer need - twkim
            weights = tf.Variable(tf.random_normal([size, num_class]), dtype=tf.float32, name='weight')
            biases = tf.Variable(tf.random_normal([num_class]), dtype=tf.float32, name='biase')

            fc = tf.matmul(input_[-1], weights) + biases
        elif learning_option.get('file_format') == "ptb": # tmp -twkim
            input_ = tf.reshape(tf.stack(axis=1, values=input_), [-1, size])
            #TODO: initializer need - twkim
            weights = tf.Variable(
                tf.truncated_normal([size,num_class], dtype=tf.float32, name='weight'))
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[num_class]), name='biase')

            #weights = tf.get_variable('weights', [size, num_class], dtype=tf.float32)
            #biases = tf.get_variable('biases', [num_class], dtype=tf.float32)

            fc = tf.matmul(input_, weights) + biases

        self.set_output('output', fc)
