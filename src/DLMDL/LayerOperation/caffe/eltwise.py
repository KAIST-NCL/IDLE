from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_eltwise(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define element-wise operation for input blobs
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
        scale = float(self.get_attr('scale', default=1.0))

        # Eltwise
        if op == 'SUM':
            op_num = 1
        elif op == 'PROD':
            op_num = 0
        elif op == 'MAX':
            op_num = 2

        eltwise = L.Eltwise(*input_, name=self.name, operation=op_num, coeff=[scale,scale])

        self.set_output('output', eltwise)
        self.set_dimension('output', indim[0])

    def run_time_operation(self, learning_option, cluster):
        pass
