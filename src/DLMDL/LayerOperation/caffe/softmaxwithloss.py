from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_softmaxwithloss(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define softmax cross entropy between logits and labels
        outputs:
            output: loss output
        """
        # get input
        logits = self.get_input('logits')
        labels = self.get_input('labels')

        loss = L.SoftmaxWithLoss(logits, labels, name=self.name)

        # set output
        self.set_output('output', loss)

    def run_time_operation(self, learning_option, cluster):
        pass
