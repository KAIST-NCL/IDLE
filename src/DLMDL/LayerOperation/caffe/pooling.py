from ..layer_operation import LayerOperation
from caffe import layers as L
import numpy as np

class op_caffe_pooling(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define pooling(max/average pooling) operation for input blob
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')
        # get attr
        # required field
        pool_type = self.get_attr('pool_type', default=None)
        if pool_type is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('type', self.name))
        kernel_size = self.get_attr('kernel_size', default=None)
        if kernel_size is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('kernel_size', self.name))

        # optional field
        padding = self.get_attr('padding', default='VALID')
        stride = self.get_attr('stride', default=1)
        engine = self.get_attr('engine', default='DEFAULT')
        global_pooling = self.get_attr('global_pooling', default=False)

        # padding
        if padding == 'SAME':
            outdim = [np.ceil(float(indim[i+2]) / float(stride)) for i in xrange(2)]
            outdim.insert(0, indim[0])
            outdim.insert(1, indim[1])
            p = [int(((outdim[i+2] - 1) * stride + kernel_size[i] - indim[i+2])/2) for i in xrange(2)]
        else:
            outdim = [np.ceil(float(indim[i+2] - kernel_size[i] + 1) / float(stride)) for i in xrange(2)]
            outdim.insert(0, indim[0])
            outdim.insert(1, indim[1])
            p = [0, 0]

        if engine == 'DEFAULT':
            engine_idx = 0
        elif engine == 'CAFFE':
            engine_idx = 1
        elif engine == 'CUDNN':
            engine_idx = 2
        else: #TODO: error handling: 'None' case
            engine_idx = 0

        # pool=0: max_pool, pool=1: avr_pool
        if pool_type == 'MAX':
            pool_type_idx = 0
        elif pool_type == 'AVG':
            pool_type_idx = 1
        else: #TODO: error handling
            pass
        pool = L.Pooling(input_, name=self.name, pool=pool_type_idx, kernel_h=kernel_size[0],
                         kernel_w=kernel_size[1], stride=stride, pad_h=p[0], pad_w=p[1],
                         engine=engine_idx, global_pooling=global_pooling)

        self.set_output('output', pool)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
