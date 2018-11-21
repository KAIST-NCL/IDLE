# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
from PIL import Image
import os
import re
import sys
import collections
import inspect

class op_tf_r_multicell(LayerOperation):

    _attributes = """[{"default": "rnn", "source": "layer", "mandatory": "tf", "name": "basic_cell"}, \
    {"default": 200, "source": "layer", "mandatory": "tf", "name": "num_units"}, \
    {"default": 0.0, "source": "layer", "mandatory": "tf", "name": "input_dropout_ratio"}, \
    {"default": 0.0, "source": "layer", "mandatory": "tf", "name": "state_dropout_ratio"}, \
    {"default": 0.0, "source": "layer", "mandatory": "tf", "name": "output_dropout_ratio"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse checkpoint which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True)

        def attn_cell():
            return tf.contrib.rnn.DropoutWrapper(
                lstm_cell(), input_keep_prob = input_keep_prob, state_keep_prob=state_keep_prob, output_keep_prob= output_keep_prob)

        def apiConstructor(image_size, is_grey):    #TODO: API constructor로 wrapping
            return

        basic_cell = self.get_attr('basic_cell')
        num_units = self.get_attr('num_units')

        size = learning_option.get('hidden_size')
        batch_size = learning_option.get('batch_size')
        num_steps = learning_option.get('num_steps')

        input_keep_prob = 1.0 - self.get_attr('input_dropout_ratio')
        state_keep_prob= 1.0 - self.get_attr('state_dropout_ratio')
        output_keep_prob = 1.0 - self.get_attr('output_dropout_ratio')

        if basic_cell == 'lstm':
            attn_cell = lstm_cell

            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(num_units)], state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)

        state = initial_state
        input_ = self.get_input('input')
        outputs = []
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(input_[:, time_step, :], state)
            outputs.append(cell_output)

        self.set_output('output', outputs)
        self.set_output('initial_state', initial_state)
        self.set_output('final_state', state)

