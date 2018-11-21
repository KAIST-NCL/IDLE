from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_lrn(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define local response normalization(LRN) operation for input tensor.
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')
        print(indim) # twtest
        # get attr
        # optional field
        local_size = self.get_attr('local_size', default=5)
        alpha = float(self.get_attr('alpha', default=1.0))
        beta = float(self.get_attr('beta', default=0.75))
        bias = float(self.get_attr('bias', default=1.0))
        norm_region = self.get_attr('norm_region', default='ACROSS_CHANNELS')
        engine = self.get_attr('engine', default='DEFAULT')

        if engine == 'DEFAULT':
            engine_idx = 0
        elif engine == 'CAFFE':
            engine_idx = 1
        elif engine == 'CUDNN':
            engine_idx = 2
        else: #TODO: error handling
            pass

        if norm_region == 'ACROSS_CHANNELS':
            norm_region_idx = 0
        elif norm_region == 'WITHIN_CHANNEL':
            norm_region_idx = 1
        else: #TODO: error handling
            pass

        lrn = L.LRN(input_, name=self.name, local_size=local_size, alpha=alpha, beta=beta,
                    norm_region=norm_region_idx, k=bias, engine=engine_idx)

        #set output dimension
        outdim = indim

        self.set_output('output', lrn)
        self.set_dimension('output', outdim)


    def run_time_operation(self, learning_option, cluster):
        pass
