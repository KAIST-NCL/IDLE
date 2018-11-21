# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L


class op_caffe_c_accuracy(LayerOperation):

    _attributes = """[\
        {"name": "topk", "mandatory": "both", "default": 1, "source": "layer"}]"""

    def compile_time_operation(self, learning_option, cluster):

        logits = self.get_input('logits')
        labels = self.get_input('labels')
        topk = self.get_attr('topk', self.topk)
        layer = L.Accuracy(logits, labels, name=self.name, accuracy_param=dict(top_k=topk))

        self.set_output('output', layer)

    def run_time_operation(self, learning_option, cluster):
        pass
