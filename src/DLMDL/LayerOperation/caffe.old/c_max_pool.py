# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import sys
from caffe import layers as L
import numpy as np


class op_caffe_c_max_pool(LayerOperation):

    _attributes = """[\
    {"source": "layer", "mandatory": "both", "name": "ksize"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "is_scale"}, \
    {"source": "layer", "mandatory": "both", "name": "stride"}, \
    {"default": "SAME", "source": "layer", "mandatory": "both", "name": "padding"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "use_global_stats"}, \
    {"default": 0.999, "source": "layer", "mandatory": "both", "name": "moving_average_fraction"}, \
    {"default": 1e-5, "source": "layer", "mandatory": "both", "name": "epsilon"}, \
    {"default": false, "source": "layer", "mandatory": "both", "name": "bias_term"}, \
    {"default": [], "source": "layer", "mandatory": "both", "name": "activation"}]"""

    def compile_time_operation(self, learning_option, cluster):
        ksize = self.get_attr('ksize')
        stride = self.get_attr('stride')
        padding = self.get_attr('padding', self.padding)

        input_ = self.get_input('input')
        name = self.name
        indim = self.get_dimension('input')
        # padding
        if padding == 'SAME':
            outdim = [np.ceil(float(indim[i]) / float(stride)) for i in xrange(2)]
            p = [int(((outdim[i] - 1) * stride + ksize - indim[i])/2) for i in xrange(2)]
        else:
            outdim = [np.ceil(float(indim[i] - ksize + 1) / float(stride)) for i in xrange(2)]
            p = [0, 0]
        # pool=0: max_pool, pool=1: avr_pool
        l_max_pool = L.Pooling(input_, name=name, pool=0, kernel_size=ksize, stride=stride, pad_h=p[0], pad_w=p[1])

        ### activation
        activation = self.get_attr('activation', self.activation)
        if len(activation) != 0:
            for act in activation:
                # relu
                if act == 'relu':
                    l_max_pool = L.ReLU(l_max_pool, name=self.name + '_relu', in_place=True)

                # # dropout
                # dropout = self.get_attr('dropout')
                # if dropout is not None:
                #     layer = L.Dropout(layer, name=self.name + '_dropout', in_place=True, dropout_ratio=dropout)

                # batch normalization
                elif act == 'batchnorm':
                    use_global_stats = self.get_attr('use_global_stats', self.use_global_stats)
                    moving_average_fraction = self.get_attr('moving_average_fraction', self.moving_average_fraction)
                    epsilon = self.get_attr('epsilon', self.epsilon)
                    l_max_pool = L.BatchNorm(l_max_pool, name=self.name + '_batchnorm',
                                                    use_global_stats=use_global_stats,
                                                    moving_average_fraction=moving_average_fraction, eps=epsilon,
                                                    in_place=True)

                    # scale
                    if self.get_attr('is_scale', self.is_scale):
                        bias_term = self.get_attr('bias_term', self.bias_term)
                        l_max_pool = L.Scale(l_max_pool, bias_term=bias_term, in_place=True)

        # TODO: output이름 DLMDL과 맞출 지 고민
        self.set_output('output', l_max_pool)
        self.set_dimension('output', outdim)
        # for output_name in self.outputs_list:
        #     self.set_output(output_name, l_max_pool)
        #     self.set_dimension(output_name, outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
