# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation


class op_caffe_c_rmspropoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.9, "source": "opt", "mandatory": "both", "name": "decay"},\
    {"default": 0.9, "source": "opt", "mandatory": "both", "name": "momentum"},\
    {"default": 1e-5, "source": "opt", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "mandatory": "tf", "name": "centered"}]"""

    def compile_time_operation(self, learning_option, cluster):
        learning_rate = learning_option.get("learning_rate")
        decay = learning_option.get("decay", self.decay)
        momentum = learning_option.get("momentum", self.momentum)
        epsilon = learning_option.get("epsilon", self.epsilon)

        for key in ['learning_rate', 'decay', 'momentum', 'epsilon']:
            try:
                del learning_option[key]
            except KeyError:
                pass
        learning_option['base_lr'] = learning_rate
        learning_option['rms_decay'] = decay
        learning_option['momentum'] = momentum
        learning_option['delta'] = epsilon
        learning_option['type'] = 'RMSProp'

    def run_time_operation(self, learning_option, cluster):
    	pass
