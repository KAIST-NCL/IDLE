# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L


class op_caffe_c_fc(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "output_shape"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"}, \
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""

    def compile_time_operation(self, learning_option, cluster):
        output_shape = self.get_attr('output_shape')
        activation = self.get_attr('activation', self.activation)

        input_ = self.get_input('input')
        # weight_filler-twkim
        weight = self.get_attr('weight', 'random_normal').lower()
        if weight == 'random_normal':
            weight_filler = {'type': 'gaussian', 'mean': 0, 'std': 0.01}
        else: # 'zeroes'
            weight_filler = {'type': 'constant'}
        # bias_filler-twkim
        bias = self.get_attr('bias', 'zeroes').lower()
        if bias == 'random_normal':
            bias_filler = {'type': 'gaussian', 'mean': 0, 'std': 0.01}
        else: # 'zeroes'
            bias_filler = {'type': 'constant'}
        layer = L.InnerProduct(input_, name=self.name, num_output=output_shape, weight_filler=weight_filler,
                               bias_filler=bias_filler)

        ### activation
        if len(activation) != 0:
            for act in activation:
                # relu
                if act == 'relu':
                    layer = L.ReLU(layer, name=self.name + '_relu', in_place=True)

                # # dropout
                # dropout = self.get_attr('dropout')
                # if dropout is not None:
                #     layer = L.Dropout(layer, name=self.name + '_dropout', in_place=True, dropout_ratio=dropout)

                # batch normalization
                elif act == 'batchnorm':
                    use_global_stats = self.get_attr('use_global_stats', self.use_global_stats)
                    moving_average_fraction = self.get_attr('moving_average_fraction', self.moving_average_Fraction)
                    epsilon = self.get_attr('epsilon', self.epsilon)
                    layer = L.BatchNorm(layer, name=self.name + '_batchnorm',
                                                    use_global_stats=use_global_stats,
                                                    moving_average_fraction=moving_average_fraction, eps=epsilon,
                                                    in_place=True)

                    # scale
                    if self.get_attr('is_scale', self.is_scale):
                        bias_term = self.get_attr('bias_term', self.bias_term)
                        layer = L.Scale(layer, bias_term=bias_term, in_place=True)

        # TODO: output이름 DLMDL과 맞출 지 고민
        self.set_output('output', layer)
        # for output_name in self.outputs_list:
        #     self.set_output(output_name, layer)


    def run_time_operation(self, learning_option, cluster):

        pass
