# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import re
import sys
import collections

class op_tf_input(LayerOperation):

    _attributes = """[{"default": 28, "source": "layer", "mandatory": "tf", "name": "num_steps"}, \
    {"default": 10, "source": "layer", "mandatory": "tf", "name": "num_class"}, \
    {"default": 128, "source": "layer", "mandatory": "tf", "name": "hidden_size"}, \
    {"default": 28, "source": "layer", "mandatory": "tf", "name": "num_units"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        batch_size = learning_option.get('batch_size')
        num_steps = self.get_attr('num_steps')
        num_class = self.get_attr('num_class')
        hidden_size = self.get_attr('hidden_size')
        num_units = self.get_attr('num_units')

        X = tf.placeholder("float", [None, num_steps, num_units])
        Y = tf.placeholder("float", [None, num_class])

        learning_option['num_steps'] = num_steps
        learning_option['num_class'] = num_class
        learning_option['hidden_size'] = hidden_size
        learning_option['num_units'] = num_units

        self.set_output('image', X)
        self.set_output('label', Y)
