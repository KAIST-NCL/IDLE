# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation


class op_caffe_c_adadeltaoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.95, "source": "opt", "mandatory": "both", "name": "rho"},\
    {"default": 1e-08, "source": "opt", "mandatory": "both", "name": "epsilon"}]"""

    def compile_time_operation(self, learning_option, cluster):
        learning_rate = learning_option.get("learning_rate")
        rho = self.get_attr("rho", self.rho)
        epsilon = self.get_attr("epsilon", self.epsilon)


        for key in ['learning_rate', 'rho', 'epsilon']:
            try:
                del learning_option[key]
            except KeyError:
                pass
        learning_option['base_lr'] = float(learning_rate)
        learning_option['momentum'] = float(rho)
        learning_option['delta'] = float(epsilon)
        learning_option['type'] = 'AdaDelta'

    def run_time_operation(self, learning_option, cluster):
    	pass
