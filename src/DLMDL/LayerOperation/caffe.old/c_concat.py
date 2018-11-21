# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L
import numpy as np


class op_caffe_c_concat(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "concat_axis"}]"""

    def compile_time_operation(self, learning_option, cluster):
        concat_axis = self.get_attr('concat_axis', 'ch')
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # Concat
        if concat_axis == 'num':
            concat_axis = 0
        elif concat_axis == 'ch':
            concat_axis = 1
        layer = L.Concat(*input_, name=self.name, axis=concat_axis)

        # TODO: output이름 DLMDL과 맞출 지고민
        self.set_output('output', layer)
        self.set_dimension('output', indim[0])
        # for output_name in self.outputs_list:
        #     print(output_name)
        #     self.set_output(output_name, layer)
        #     self.set_dimension(output_name, indim[0])

    def run_time_operation(self, learning_option, cluster):
        pass
