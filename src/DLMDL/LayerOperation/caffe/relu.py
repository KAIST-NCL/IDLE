from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_relu(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define rectified-linear unit(ReLU) operation for input blob
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        slope = float(self.get_attr('slope', default=0.0))
        engine = self.get_attr('engine', default='DEFAULT')

        if engine == 'DEFAULT':
            engine_idx = 0
        elif engine == 'CAFFE':
            engine_idx = 1
        elif engine == 'CUDNN':
            engine_idx = 2
        else:  # TODO: error handling: 'None' case
            engine_idx = 0

        relu = L.ReLU(input_, name=self.name, negative_slope=slope, engine=engine_idx)

        #set output dimension
        outdim = indim

        self.set_output('output', relu)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
