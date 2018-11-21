# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
from PIL import Image
import os
import re
import sys
import collections

'''
input layer for glaucoma images from ETRI project
image size is fixed with [224,224,3]
'''

class op_tf_galucoma_input(LayerOperation):

    _attributes = """[]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def apiConstructor():
            images_placeholder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='images')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels')
            return images_placeholder, labels_placeholder
            
        device = self.get_attr('device')
        num = re.sub('[^0-9]','',cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num),'')
        '''
        if learning_option.get("parallel", None) != "DP":
            with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    images_placeholder, labels_placeholder = apiConstructor()
        else:
            images_placeholder, labels_placeholder = apiConstructor()
        '''
        images_placeholder, labels_placeholder = apiConstructor()
        

        outdim = list(images_placeholder.get_shape()[i].value for i in xrange(len(images_placeholder.get_shape())))
        self.set_output('image', images_placeholder)
        self.set_output('label', labels_placeholder)
        self.set_dimension('image', outdim)
