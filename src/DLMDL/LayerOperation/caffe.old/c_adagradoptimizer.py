# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation


class op_caffe_c_adagradoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.1, "source": "opt", "mandatory": "tf", "name": "initial_accumulator_value"}]"""

    def compile_time_operation(self, learning_option, cluster):
        learning_rate = learning_option.get("learning_rate")
        #initial_accumulator_value = learning_option.get("initial_accumulator_value", self.initial_accumulator_value)

        for key in ['learning_rate', 'initial_accumulator_value']:
            try:
                del learning_option[key]
            except KeyError:
                pass
        learning_option['base_lr'] = float(learning_rate)
        learning_option['type'] = 'AdaGrad'

    def run_time_operation(self, learning_option, cluster):
    	pass
