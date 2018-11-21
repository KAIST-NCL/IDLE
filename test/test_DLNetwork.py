import unittest
from src.Serializer.serializer import JsonSerialize
from src.DLMDL.DLNetwork import DLNetwork


class DLNetworkTestCase(unittest.TestCase):
    def setUp(self):
        serializer = JsonSerialize()
        serializer.parse('../data/unit_test_input.json')
        self.network = DLNetwork(serializer)

    def test_forward(self):
        self.assertEqual(self.network.layers["Input"].get_next_layer('Input.train_image')[1], self.network.layers["Conv1"])

    def test_backward(self):
        self.assertEqual(self.network.layers["Conv1"].get_prev_layer('test', 'Input.test_label'), self.network.layers["Input"])

    def test_object_pass(self):
        self.network.layers["Input"].set_op_output('test_image', 'Conv1_output_obj')
        self.assertEqual(self.network.layers["Conv1"].get_op_input('test')[2], 'Conv1_output_obj')

if __name__ == '__main__':
    unittest.main()
