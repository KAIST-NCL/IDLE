# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L
import numpy as np


class op_caffe_c_drop(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "dropout_ratio"}]"""


    def compile_time_operation(self, learning_option, cluster):
        dropout_ratio = self.get_attr('dropout_ratio')
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # Dropout
        layer = L.Pooling(input_, name=self.name, dropout_ratio=dropout_ratio)

        # TODO: output이름 DLMDL과 맞출 지 고민
        self.set_output('output', layer)
        self.set_dimension('output', indim)
        # for output_name in self.outputs_list:
        #     self.set_output(output_name, layer)
        #     self.set_dimension(output_name, indim)

    def run_time_operation(self, learning_option, cluster):

        pass
