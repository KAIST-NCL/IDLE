# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation


class op_caffe_c_nesterovoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.95, "source": "opt", "mandatory": "caffe", "name": "momentum"}]"""

    def compile_time_operation(self, learning_option, cluster):
        learning_rate = learning_option.get("learning_rate")
        momentum = learning_option.get("momentum", self.momentum)

        for key in ['learning_rate', 'momentum']:
            try:
                del learning_option[key]
            except KeyError:
                pass
        learning_option['base_lr'] = learning_rate
        learning_option['momentum'] = momentum
        learning_option['type'] = 'Nesterov'

    def run_time_operation(self, learning_option, cluster):
    	pass
