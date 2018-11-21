from ..layer_operation import LayerOperation
from caffe import layers as L

class op_caffe_accuracy(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define accuracy based on logits and labels
        """
        # get input
        logits = self.get_input('logits')
        labels = self.get_input('labels')

        # get attr
        # optional field
        topk = self.get_attr('topk', default=1)

        accuracy = L.Accuracy(logits, labels, name=self.name, accuracy_param=dict(top_k=topk))

        self.set_output('output', accuracy)

    def run_time_operation(self, learning_option, cluster):
        pass