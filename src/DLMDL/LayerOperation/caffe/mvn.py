from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_mvn(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define mean-variance normalization(MVN) operation for input tensor.
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        normalize_variance = self.get_attr('normalize_variance', default=True)
        across_channels = self.get_attr('across_channels', default=False)
        eps = float(self.get_attr('epsilon', default=10**-9))

        mvn = L.MVN(input_, name=self.name, normalize_variance=normalize_variance,
                    across_channels=across_channels, epsilon=eps)

        #set output dimension
        outdim = indim

        self.set_output('output', mvn)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
