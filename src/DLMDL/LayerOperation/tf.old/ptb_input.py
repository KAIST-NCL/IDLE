# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
from PIL import Image
import os
import re
import sys
import collections

class op_tf_ptb_input(LayerOperation):

    _attributes = """[{"default": 20, "source": "layer", "mandatory": "tf", "name": "num_steps"}, \
    {"default": 10000, "source": "layer", "mandatory": "tf", "name": "num_class"}, \
    {"default": 200, "source": "layer", "mandatory": "tf", "name": "hidden_size"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):

        num_steps = self.get_attr('num_steps') # [mandatory attr] input text length at once
        num_class = self.get_attr('num_class') # [mandatory attr] total number of words
        hidden_size = self.get_attr('hidden_size') # [mandatory attr] word embedding size 
        
        # set to learning option
        learning_option['num_steps'] = num_steps
        learning_option['num_class'] = num_class
        learning_option['hidden_size'] = hidden_size
        
        learning_option['classes'] = hidden_size
        
        # dataset or test procedure
        is_train = tf.placeholder_with_default(True, shape=())
        learning_option['is_train'] = is_train

        # get or create embedding variable
        embedding = tf.get_variable(name='embedding', 
            shape=[num_class, hidden_size], dtype=tf.float32)

        # input data & target placeholder
        input_data = tf.placeholder(dtype=tf.int32, shape=(None, num_steps), name='ptb_input')
        targets = tf.placeholder(dtype=tf.int32, shape=(None, num_steps), name='ptb_targets')
        
        learning_option['data_placeholder'] = input_data
        learning_option['targets_placeholder'] = targets

        input_data = tf.nn.embedding_lookup(embedding, input_data)

        # set layer output
        self.set_output('text', input_data)
        self.set_output('targets', targets)

