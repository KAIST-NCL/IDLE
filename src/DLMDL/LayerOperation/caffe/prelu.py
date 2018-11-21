from ..layer_operation import LayerOperation
from caffe import layers as L
from initializer import get_initializer
from regularizer import get_regularizer

class op_caffe_prelu(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define parame rectified-linear unit(ReLU) operation for input tensor
        It follows: f(x) = alpha * x for x < 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        initializer = self.get_attr('initializer', default={'weight': {}, 'bias': {}})  # default will set later
        regularizer = self.get_attr('regularizer', default={})  # default will set later
        ch_shared = self.get_attr('channel_shared', default=False)

        # get weight for convolution
        alpha_init = get_initializer(initializer.get('weight'), is_bias=False)
        alpha_reg, alpha_reg_type = get_regularizer(regularizer, is_bias=False)

        # check regularizer type
        tmp_reg = learning_option.get('caffe_reg_type')
        if tmp_reg is None:
            learning_option['caffe_reg_type'] = alpha_reg_type
        else:
            if tmp_reg != alpha_reg_type:
                raise Exception('[DLMDL ERROR]: In caffe, regularizer type of all layers must be equal')

        prelu = L.PReLU(input_, name=self.name, weight_filler=alpha_init,
                       channel_shared=ch_shared, param=[alpha_reg])

        #set output dimension
        outdim = indim

        self.set_output('output', prelu)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
