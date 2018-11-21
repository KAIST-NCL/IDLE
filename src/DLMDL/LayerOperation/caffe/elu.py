from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_elu(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define exponential-linear unit(ELU) operation for input tensor
        Computes exponential linear: exp(features) - 1 if < 0, features otherwise.
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        slope = float(self.get_attr('slope', default=1.0))

        elu = L.ELU(input_, name=self.name, alpha=slope)

        #set output dimension
        outdim = indim

        self.set_output('output', elu)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
