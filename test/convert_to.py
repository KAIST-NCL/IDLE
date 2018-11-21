import tensorflow as tf
import time
import socket
import sys
import src.util.codegenUtil as gen
import inspect
import importlib
import codecs
import json
from collections import OrderedDict
import numpy as np
import os
from PIL import Image

import cv2
import matplotlib.pyplot as plt
"""
"image_size": [256, 256]
"""
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class Dataset:
    def __init__(self, img_path, file_format, image_size, label_path, batch_size):
        self.index_in_epoch = 0 
        self.epochs_completed = 0
        self.file_format = file_format
        self.img_path = img_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.data = []
        self.label = []
	pass
    """
    def _int64_feature(self, value):
        return tf.datasets.Feature(int64_list=tf.datasets.Int64List(value=[value]))


    def _bytes_feature(self, value):
        return tf.datasets.Feature(bytes_list=tf.datasets.BytesList(value=[value]))
    """
    def load_image(self, addr):
        img = cv2.imread(addr)
        img = cv2.resize(img, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def label_read(self, batch_filenames):
        for file in batch_filenames:
           #print(os.path.splitext(file)[0])
            with open(self.img_path + '/' + self.label_path) as fp:
                for line in fp:
                    if os.path.splitext(file)[0] in line:
                        # int convert is conflict?
                        self.label.append(line.split(' ')[-1].rstrip())
                        break

    def convert_to(self):
        filenames = os.listdir(self.img_path)

        # make dataset which consists of only specific-formated files (e.g., '.jpg')
        for filename in filenames:
            ext = os.path.splitext(str(filename))[1] # filename == w/o path, ONLY name and extension
            if ext != '.' + self.file_format:
                filenames.remove(filename)
                # filenames consists of all images (e.g., filename.extension), len(filenames) == # of dataset
   
        name = os.path.join(self.img_path, "training_data" + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(name)

        self.label_read(filenames)
        
        rows = self.image_size[0]
        cols = self.image_size[1]

        for index, name in enumerate(filenames):  #filenames is list
           img = self.load_image(name)
           print index, name
           self.data.append(img)
           depth = img.shape[2]
           image_raw = self.data[index].tostring()
           example = tf.train.Example(features=tf.train.Features(feature={ 'height': _int64_feature(rows), 'width': _int64_feature(cols), 'depth': _int64_feature(depth), 'label': _int64_feature(int(self.label[index])), 'image_raw': _bytes_feature(image_raw)})) 
           writer.write(example.SerializeToString())

    def read_tfrecord(self):
        data_path = os.path.join(self.img_path, "training_data" + '.tfrecords')
        print data_path
        feature={ 'height':tf.FixedLenFeature([], tf.int64) , 'width': tf.FixedLenFeature([], tf.int64), 'depth': tf.FixedLenFeature([], tf.int64), 'label': tf.FixedLenFeature([], tf.int64), 'image_raw': tf.FixedLenFeature([], tf.string)}

        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue) 
        features = tf.parse_single_example(serialized_example, features=feature)

        image = tf.decode_raw(features['image_raw'], tf.float32)

        label = tf.cast(features['label'], tf.int32)
        # How to find DEPTH info?
        image = tf.reshape(image, [self.image_size[0], self.image_size[1], 3])

        images, labels = tf.train.shuffle_batch([image, label], batch_size=self.batch_size, capacity=30, num_threads=1, min_after_dequeue=10)
        print images.shape

if __name__ == "__main__":
    data = Dataset('../data/datasets/flowers', 'jpg', [256, 256], 'test_labeling.txt', 10)
    #data.convert_to()
    data.read_tfrecord()
