from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_sigmoid(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define sigmoid operation for input tensor
        Computes sigmoid of x element-wise. Specifically, y = 1 / (1 + exp(-x)).
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        engine = self.get_attr('engine', default='DEFAULT')

        if engine == 'DEFAULT':
            engine_idx = 0
        elif engine == 'CAFFE':
            engine_idx = 1
        elif engine == 'CUDNN':
            engine_idx = 2
        else: #TODO: error handling
            pass

        sigmoid = L.Sigmoid(input_, name=self.name, engine=engine_idx)

        #set output dimension
        outdim = indim

        self.set_output('output', sigmoid)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
