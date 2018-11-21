from ..layer_operation import LayerOperation
from caffe import layers as L


class op_caffe_reduction(LayerOperation):
    _attributes = """[]"""  # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define reduction operation for input blob
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        op = self.get_attr('operation', default=None)
        if op is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('op', self.name))

        # optional field
        axis = self.get_attr('axis', default=None)
        scale = float(self.get_attr('scale', default=1.0))

        # get output dimension
        if axis == len(indim):
            indim.pop()
            outdim = indim
        else:
            outdim = indim
            outdim[axis] = 1

        reduction = L.Reduction(input_, name=self.name, operation=op, axis=axis, coeff=scale)

        # set output
        self.set_output('output', reduction)
        self.set_dimension('output', outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
