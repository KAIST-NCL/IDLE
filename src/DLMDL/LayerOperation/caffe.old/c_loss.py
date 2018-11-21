# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L


class op_caffe_c_loss(LayerOperation):

    _attributes = """[]"""

    def compile_time_operation(self, learning_option, cluster):

        logits = self.get_input('logits')
        labels = self.get_input('labels')
        layer = L.SoftmaxWithLoss(logits, labels, name=self.name)
        self.set_output('output', layer)
        # for output_name in self.outputs_list:
        #     self.set_output(output_name, layer)

    def run_time_operation(self, learning_option, cluster):

        pass
