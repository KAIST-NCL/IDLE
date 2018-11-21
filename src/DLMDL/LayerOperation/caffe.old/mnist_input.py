# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import random
import csv
import sys
import caffe
from caffe import layers as L

from src.DLMDL.caffe_adaptor import tempNet # TODO: DLMDL에 solver 따로 만들거나 변경시 재검토 필요

class op_caffe_mnist_input(LayerOperation):

    _attributes = """[{"default": "datasets", "source": "opt", "mandatory": "all", "name": "option"}, {"default": "jpg", "source": "opt", "mandatory": "all", "name": "file_format"}, {"source": "opt", "mandatory": "all", "name": "data_path"}, {"source": "opt", "mandatory": "all", "name": "label_path"}, {"source": "opt", "mandatory": "all", "name": "batch_size"}, {"source": "opt", "mandatory": "all", "name": "iteration"}, {"source": "layer", "mandatory": "all", "name": "image_size"}, {"source": "layer", "mandatory": "all", "name": "output_shape"}]"""

    def compile_time_operation(self, learning_option, cluster):
        option = learning_option.get("option", self.option)
        file_format = learning_option.get("file_format", self.file_format)
        data_path = learning_option.get("data_path", self.data_path)
        label_path = learning_option.get("label_path", self.label_path)
        batch_size = learning_option.get("batch_size", self.batch_size)
        iteration = learning_option.get("iteration", self.iteration)
        image_size = self.image_size
        output_shape = self.output_shape
        
        # for shmcaffe
        #learning_option["move_rate"] = learning_option.get("move_rate", 0.2)
        #learning_option["tau"] = learning_option.get("tau", 1)

        # Phase checkpoint setting, PHASE: 0 for trian, 1 for test
        isTrainTest = 0
        if option.lower() == "test":
            temp_include = dict(phase=caffe.TEST)
            data_path = learning_option.get("test_data_path", data_path)
            test_label_path = learning_option.get("test_label_path", label_path)
            batch_size = learning_option.get("test_batch_size", batch_size)
        elif option.lower() == "datasets":
            temp_include = dict(phase=caffe.TRAIN)
        elif option.lower() == "train_test":
            temp_include = dict(phase=caffe.TRAIN)
            isTrainTest = 1
        else:
            temp_include = dict(phase=caffe.TRAIN)

        # DB Data
        if file_format.lower() in ["lmdb", "leveldb"]:
            print('twkim')
            # Backend checkpoint setting, default value 0 (leveldb) for backend
            # Data layer setting
            image, label = L.Data(name=self.name, source=data_path,
                             batch_size=batch_size, backend=(0 if file_format.lower()=="leveldb" else 1), include=temp_include, ntop=2)

            if isTrainTest == 1:
                data_path = learning_option.get("test_data_path", data_path)
                batch_size = learning_option.get("test_batch_size", batch_size)
                temp_image, temp_label = L.Data(name=self.name, source=data_path,
                                                batch_size=batch_size,
                                                backend=(0 if file_format.lower() == "leveldb" else 1),
                                                include=dict(phase=caffe.TEST), ntop=2)
                setattr(tempNet, str(self.name) + '.image', temp_image)
                setattr(tempNet, str(self.name) + '.label', temp_label)
        else: # mnist dataset only support lmdb or leveldb in caffe 
            pass

        # Record the layer output information
        print('twkim')
        print(iteration)
        self.set_output('image', image)
        self.set_output('label', label)
        self.set_dimension('image', image_size)

        try:
            if isTrainTest != 1:
                del learning_option['option']
            del learning_option['file_format']
            del learning_option['data_path']
            del learning_option['label_path']
            del learning_option['batch_size']
            del learning_option['iteration']
            print('twkim')
            learning_option['max_iter'] = iteration
        except KeyError:
            pass

        try:
            del learning_option['test_data_path']
            del learning_option['test_label_path']
            del learning_option['test_batch_size']
            learning_option['test_iter'] = learning_option.get("test_iteration", 100)
            del learning_option['test_iteration']
        except KeyError:
            pass

    def run_time_operation(self, learning_option, cluster):
        pass
