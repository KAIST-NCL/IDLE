# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L
import numpy as np


class op_caffe_c_conv(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "filter"}, \
    {"source": "layer", "mandatory": "both", "name": "stride"}, \
    {"default": "NHWC", "source": "layer", "mandatory": "tf", "name": "data_format"},\
    {"default": true, "source": "layer", "mandatory": "tf", "name": "use_cudnn_on_gpu"},\
    {"default": "SAME", "source": "layer", "mandatory": "both", "name": "padding"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"},\
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""

    def compile_time_operation(self, learning_option, cluster):
        filter = self.get_attr('filter')
        stride = self.get_attr('stride')
        padding = self.get_attr('padding', self.padding)
        activation = self.get_attr('activation', self.activation)

        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # padding
        if padding == 'SAME':
            outdim = [np.ceil(float(indim[i]) / float(stride)) for i in xrange(2)]
            p = [int(((outdim[i] - 1) * stride + filter[i] - indim[i])/2) for i in xrange(2)]
        else:
            outdim = [np.ceril(float(indim[i] - filter[i] + 1) / float(stride)) for i in xrange(2)]
            p = [0, 0]

        # weight_filler-twkim
        weight = self.get_attr('weight', 'random_normal').lower()
        if weight == 'random_normal':
            weight_filler = {'type': 'gaussian', 'mean': 0, 'std': 0.01}
        else: # 'zeroes'-twkim
            weight_filler = {'type': 'constant'}
        # bias_filler-twkim
        bias = self.get_attr('bias', 'zeroes').lower()
        if bias == 'random_normal':
            bias_filler = {'type': 'gaussian', 'mean': 0, 'std': 0.01}
        else: # '0.1 values' -twkim
            bias_filler = {'type': 'constant', 'value': 0.1}
        layer = L.Convolution(input_, name=self.name, kernel_h=filter[0], kernel_w=filter[1], num_output=filter[3], stride=stride,
                               pad_h=p[0], pad_w=p[1], weight_filler=weight_filler, bias_filler=bias_filler)

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
                    moving_average_fraction = self.get_attr('moving_average_fraction', self.moving_average_fraction)
                    epsilon = self.get_attr('epsilon', self.epsilon)
                    layer = L.BatchNorm(layer, name=self.name + '_batchnorm', use_global_stats=use_global_stats,
                                                  moving_average_fraction=moving_average_fraction, eps=epsilon, in_place=True)

                    # scale
                    if self.get_attr('is_scale', self.is_scale):
                        bias_term = self.get_attr('bias_term', self.bias_term)
                        layer = L.Scale(layer, bias_term=bias_term, in_place=True)


        # TODO: output이름 DLMDL과 맞출 지 고민
        self.set_output('output', layer)
        self.set_dimension('output', outdim)
        # for output_name in self.outputs_list:
        #     self.set_output(output_name, layer)
        #     self.set_dimension(output_name, outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
