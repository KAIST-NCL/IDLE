from ..layer_operation import LayerOperation
from caffe import layers as L
import numpy as np
from initializer import get_initializer
from regularizer import get_regularizer

class op_caffe_conv(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define convolution operation for input blob
        """
        #get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        #get attr
        #required field
        kernel_size = self.get_attr('kernel_size', default=None)
        if kernel_size is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('kernel_size', self.name))
        num_output = self.get_attr('num_output', default=None)
        if num_output is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_output', self.name))

        #optional field
        padding = self.get_attr('padding', default='VALID')
        stride = self.get_attr('stride', default=1)
        bias_term = self.get_attr('bias_term', default=True)
        initializer = self.get_attr('initializer', default={'weight': {}, 'bias': {}})  # default will set later
        regularizer = self.get_attr('regularizer', default={})  # default will set later
        group = self.get_attr('group', default=1)

        # get weight for convolution
        weight_init = get_initializer(initializer.get('weight'), is_bias=False)
        weight_reg, weight_reg_type = get_regularizer(regularizer, is_bias=False)

        # if bias_term is True, add bias term to convolution output
        if bias_term:
            bias_init = get_initializer(initializer.get('bias'), is_bias=True)
            bias_reg, bias_reg_type = get_regularizer(regularizer, is_bias=True)
        else:
            bias_init = None
            bias_reg = None
            bias_reg_type = None

        # check regularizer type
        tmp_reg = learning_option.get('caffe_reg_type')
        if tmp_reg is None:
            learning_option['caffe_reg_type'] = weight_reg_type
        else:
            if tmp_reg != weight_reg_type or tmp_reg !=  bias_reg_type:
                raise Exception('[DLMDL ERROR]: In caffe, regularizer type of all layers must be equal')

        # padding
        if padding == 'SAME':
            outdim = [np.ceil(float(indim[i+2]) / float(stride)) for i in xrange(2)]
            outdim.insert(0, indim[0])
            outdim.insert(1, num_output)
            p = [int(((outdim[i+2] - 1) * stride + kernel_size[i] - indim[i+2])/2) for i in xrange(2)]
        else:
            outdim = [np.ceil(float(indim[i+2] - kernel_size[i] + 1) / float(stride)) for i in xrange(2)]
            outdim.insert(0, indim[0])
            outdim.insert(1, num_output)
            p = [0, 0]

        conv = L.Convolution(input_, name=self.name, kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                              num_output=num_output, stride=stride, group=group, pad_h=p[0], pad_w=p[1],
                              weight_filler=weight_init, bias_filler=bias_init, param=[weight_reg, bias_reg])


        self.set_output('output', conv)
        self.set_dimension('output', outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
