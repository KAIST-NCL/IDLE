from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_reshape(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define reshape operation for input blob
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        shape = self.get_attr('shape', default=None)
        if shape is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('shape', self.name))
        # in Caffe
        # 0: means 'copy the respective dimension of the bottom layer'
        # -1: stands for 'infer this from the other dimensions'

        reshape = L.Reshape(input_, name=self.name ,reshape=shape)

        # dimension calculation
        outdim=[]
        for idx, dim in enumerate(shape):
            if dim == 0:
                outdim[idx] = indim[idx]
            elif dim == -1:
                tmp_dim = 1
                for num in range(idx, len(indim)):
                    tmp_dim *= indim[num]
                outdim[idx] = tmp_dim
            else:
                outdim[idx] = dim

        self.set_output('output', reshape)
        self.set_dimension('output', outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
