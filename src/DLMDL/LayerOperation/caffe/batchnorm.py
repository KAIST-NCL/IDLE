from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_batchnorm(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define batch normalization operation for input blob
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # optional field
        eps = float(self.get_attr('epsilon', default=10 ** -5))
        moving_avg = float(self.get_attr('moving_average', default=0.999))
        use_global_stats = self.get_attr('use_globl_stats', default=False)

        batchnorm = L.BatchNorm(input_, name=self.name , use_global_stats=use_global_stats,
                            moving_average_fraction=moving_avg, eps=eps)

        self.set_output('output', batchnorm)
        self.set_dimension('output', indim)

    def run_time_operation(self, learning_option, cluster):
        pass
