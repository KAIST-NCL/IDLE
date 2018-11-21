from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_database_input(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define input placeholder for database(TFRECORD) input.
        outputs:
            data: data placeholder
            label: label data placeholder
        """
        # TODO: implement
        pass



