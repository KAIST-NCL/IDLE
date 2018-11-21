# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
from PIL import Image
import os
import re
import sys
import collections

class op_tf_jpeg_input(LayerOperation):

    _attributes = """[{"default": 20, "source": "layer", "mandatory": "tf", "name": "num_steps"}, \
    {"default": 10000, "source": "layer", "mandatory": "tf", "name": "vocab_size"}, \
    {"default": 200, "source": "layer", "mandatory": "tf", "name": "hidden_size"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor(image_size):
            #images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size[0], image_size[1], 1 if is_grey == True else 3))
            images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size[0], image_size[1], None))
            labels_placeholder = tf.placeholder(tf.int64, shape=(None))
            return images_placeholder, labels_placeholder

       
        """
        input layer operation returns random batch from input image
        :return: [image, label]
       """
        file_format = learning_option.get("file_format")
        data_path = learning_option.get("data_path")
        image_size = self.get_attr('image_size')
        #is_grey = is_grey_scale(data_path, file_format)
        learning_option['image_size'] = image_size
        
        device = self.get_attr('device')
        num = re.sub('[^0-9]','',cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num),'')

        if learning_option.get("parallel", None) != "DP":
            with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    images_placeholder, labels_placeholder = apiConstructor(image_size)
        else:
            images_placeholder, labels_placeholder = apiConstructor(image_size)

        outdim = list(images_placeholder.get_shape()[i].value for i in xrange(len(images_placeholder.get_shape())))
        self.set_output('image', images_placeholder)
        self.set_output('label', labels_placeholder)
        self.set_dimension('image', outdim)
