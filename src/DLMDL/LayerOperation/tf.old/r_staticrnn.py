# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
import os
import re
import sys
import collections
import inspect

class op_tf_r_staticrnn(LayerOperation):

    _attributes = """[{"default": "rnn", "source": "layer", "mandatory": "tf", "name": "basic_cell"}, \
    {"default": 0.0, "source": "layer", "mandatory": "tf", "name": "forget_bias"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):

        basic_cell = self.get_attr('basic_cell')
        forget_bias = self.get_attr('forget_bias')
        input_ = self.get_input('input')

        size = learning_option.get('hidden_size')
        num_steps = learning_option.get('num_steps')

        input_ = tf.unstack(input_, num_steps, 1)

        if basic_cell == 'lstm':
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=forget_bias)
            outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, input_, dtype=tf.float32)

        self.set_output('output', outputs)
        self.set_output('states', states)
