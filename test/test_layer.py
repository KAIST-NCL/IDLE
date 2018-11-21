import unittest

from src.DLMDL.layer import *
from src.Serializer.serializer import *


class LayerTestCase(unittest.TestCase):

    def setUp(self):
        serializer = JsonSerialize()
        serializer.parse('../data/unit_test_input.json')
        layers_def = serializer.get_network()
        # for layer_def in layers_def:
        self.layer1 = Layer(resolver=serializer.new_layer_resolver(layers_def[0]))
        self.layer2 = Layer(resolver=serializer.new_layer_resolver(layers_def[1]))
        self.layer2.add_link('input', self.layer1)
        self.layer2.add_link('test', self.layer1)

    def tearDown(self):
        self.layer1 = None
        self.layer2 = None

    def test_forward(self):
        self.assertEqual(self.layer1.get_next_layer('Input.train_image')[1], self.layer2)

    def test_backward(self):
        self.assertEqual(self.layer2.get_prev_layer('test', 'Input.test_label'), self.layer1)

    def test_object_pass(self):
        self.layer1.set_op_output('test_image', 'Conv1_output_obj')
        self.assertEqual(self.layer2.get_op_input('test')[2], 'Conv1_output_obj')

    def test_operation(self):
        print(self.layer1.op.inspect_operation())
        print(self.layer2.op.inspect_operation())

        self.layer1.operate()
        self.layer2.operate()


if __name__ == '__main__':
    unittest.main()
