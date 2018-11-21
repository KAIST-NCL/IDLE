from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_split(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define split operation for input blobs
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        # WARNING: size_split is only required in Caffe, not TF or MXNet
        size_split = self.get_attr('size_split', default=None)
        if size_split is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('size_split', self.name))
        slice_point = []
        for idx, val in enumerate(size_split):
            if idx == 0:
                slice_point.insert(idx, val)
            else:
                slice_point.insert(idx, val + size_split[idx])


        # optional field
        axis = self.get_attr('axis', default=0)
        slice_param = {'axis': axis, 'slice_point':slice_point}

        # get output dimension
        outdim = []
        ntop = len(size_split)
        for i in range(ntop):
            outdim.insert(i, [])
            for j in range(len(indim)):
                if j != axis:
                    outdim[i].insert(j, indim[j])
                else:
                    outdim[i].insert(j, size_split[j])

        slice = L.Slice(input_, name=self.name, slice_param=slice_param, ntop=ntop)

        # set output
        for idx, val in enumerate(slice):
            self.set_output('output{0}'.fomrmat(idx), val)
            self.set_dimension('output{0}'.format(idx), outdim[idx])

    def run_time_operation(self, learning_option, cluster):
        pass
