from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_dropout(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
       define dropout operation for input tensor
       """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        dropout_ratio = float(self.get_attr('dropout_ratio', default=0.0))
        if dropout_ratio is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('dropout_ratio', self.name))

        dropout = L.Dropout(input_, name=self.name, dropout_ratio=dropout_ratio)

        self.set_output('output', dropout)
        self.set_dimension('output', indim)

    def run_time_operation(self, learning_option, cluster):
        pass
