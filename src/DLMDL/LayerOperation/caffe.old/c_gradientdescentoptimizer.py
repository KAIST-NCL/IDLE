# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation


class op_caffe_c_gradientdescentoptimizer(LayerOperation):

    _attributes = """[{"default": 0.001, "source": "opt", "mandatory": "both", "name": "learning_rate"}, {"default": false, "mandatory": "tf", "name": "use_locking"}]"""

    def compile_time_operation(self, learning_option, cluster):
        learning_rate = learning_option.get("learning_rate", self.learning_rate)

        for key in ['learning_rate', 'use_locking']:
            try:
                del learning_option[key]
            except KeyError:
                pass
        learning_option['base_lr'] = learning_rate
        learning_option['type'] = 'SGD'

    def run_time_operation(self, learning_option, cluster):
    	pass
