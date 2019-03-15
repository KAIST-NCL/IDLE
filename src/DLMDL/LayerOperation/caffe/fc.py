from ..layer_operation import LayerOperation
from caffe import layers as L
from initializer import get_initializer
from regularizer import get_regularizer

class op_caffe_fc(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define fully-connected(FC) operation for input tensor.
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        num_output = self.get_attr('num_output', default=None)
        if num_output is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_output', self.name))

        # optional field
        bias_term = self.get_attr('bias_term', default=True)
        initializer = self.get_attr('initializer', default={'weight': {}, 'bias': {}})  # default will set later
        regularizer = self.get_attr('regularizer', default={})  # default will set later

        # get weight for convolution
        weight_init = get_initializer(initializer.get('weight'), is_bias=False)
        weight_reg, weight_reg_type = get_regularizer(regularizer, is_bias=False)
        decay_mul = [weight_reg]
        # if bias_term is True, add bias term to convolution output
        if bias_term:
            bias_init = get_initializer(initializer.get('bias'), is_bias=True)
            bias_reg, bias_reg_type = get_regularizer(regularizer, is_bias=True)
            decay_mul.append(bias_reg)
        else:
            bias_init = {}

        # check regularizer type
        tmp_reg = learning_option.get('caffe_reg_type')
        if tmp_reg is None:
            learning_option['caffe_reg_type'] = weight_reg_type
        else:
            if tmp_reg != weight_reg_type:
                raise Exception('[DLMDL ERROR]: In caffe, regularizer type of all layers must be equal')

        fc = L.InnerProduct(input_, name=self.name, num_output=num_output,
                            weight_filler=weight_init, bias_filler=bias_init,
                            param=decay_mul)

        outdim = [indim[0], num_output]

        self.set_output('output', fc)
        self.set_dimension('output', outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
