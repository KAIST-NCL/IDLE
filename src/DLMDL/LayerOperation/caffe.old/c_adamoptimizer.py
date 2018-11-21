# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation


class op_caffe_c_adamoptimizer(LayerOperation):

    _attributes = """[\
    {"default": 0.9, "source": "opt", "mandatory": "both", "name": "beta1"}, \
    {"default": 0.999, "source": "opt", "mandatory": "both", "name": "beta2"}, \
    {"default": 1e-08, "source": "opt", "mandatory": "both", "name": "epsilon"}]"""

    def compile_time_operation(self, learning_option, cluster):
        learning_rate = learning_option.get("learning_rate")
        beta1 = self.get_attr("beta1", self.beta1)
        beta2 = self.get_attr("beta2", self.beta2)
        epsilon = self.get_attr("epsilon", self.epsilon)

        # default values
        # sol.net = "/home/ncl/caffe/examples/msjeon/dlmdl2caffe/dlmdl2caffe.prototxt"
        # sol.lr_policy = "fixed"
        # sol.display = 50
        #
        # # specified values
        # sol.base_lr = float(layer["attributes"]["learning_rate"])
        #
        # if "beta1" in layer["attributes"]:
        #     sol.momentum = float(layer["attributes"]["beta1"])
        #
        # if "beta2" in layer["attributes"]:
        #     sol.momentum2 = float(layer["attributes"]["bta2"])
        #
        # ### epsilon, train_iteration(input layer), test_iteration, test_interval????????????
        for key in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
            try:
                del learning_option[key]
            except KeyError:
                pass
        learning_option['base_lr'] = float(learning_rate)
        learning_option['momentum'] = float(beta1)
        learning_option['momentum2'] = float(beta2)
        learning_option['delta'] = float(epsilon)
        learning_option['type'] = 'Adam'

    def run_time_operation(self, learning_option, cluster):

        pass
