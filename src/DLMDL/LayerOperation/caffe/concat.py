from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_concat(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define concat operation for input blobs
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        axis = self.get_attr('axis', default=None)
        if axis is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('axis', self.name))

        outdim = []
        for i in range(len(indim[0])):
            tmp = 0
            if i == axis:
                for j in range(len(indim)):
                    tmp += indim[j][i]
            else:
                tmp = indim[0][i]
            outdim.insert(i, tmp)

        concat = L.Concat(input_, name=self.name, axis=axis)

        self.set_output('output', concat)
        self.set_dimension('output', outdim)

    def run_time_operation(self, learning_option, cluster):
        pass
