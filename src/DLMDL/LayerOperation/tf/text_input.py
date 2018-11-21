from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_text_input(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define input placeholder for text input.
        outputs:
            text: text data placeholder
            targets: target data placeholder
        """
        # TODO: implement
        pass


