# -*- coding: utf-8 -*-

import unittest
from src.DLMDL.inout import *
from src.DLMDL.layer import Layer


class InOutTestCase(unittest.TestCase):
    # TODO: 코드 수정 필요
    def setUp(self):
        layer1 = Layer(name="Input", inputs={}, outputs=[], type='input')
        layer2 = Layer(name="Conv", inputs={}, outputs=[], type='conv')

        self.output = Out(layer1, [
        "train_image",
        "train_label",
        "test_image",
        "test_label"
        ])

        self.input = In(layer2, {
        "input": [
          "Input.train_image",
          "Input.train_label"
        ],
        "test": [
          "Input.train_image",
          "Input.train_label",
          "Input.test_image",
          "Input.test_label"
        ]
        })

        self.input.add_link('test', self.output)
        self.output.set_obj("test_image", 'prev_output_data')

    def tearDown(self):
        self.output = None
        self.input = None

    def test_get_obj(self):
        self.assertEqual(self.input.get_obj('test')[2], 'prev_output_data')

    def test_invalid_input(self):
        self.assertRaises(KeyError, self.input.get_obj, 'test_invalid_input')

    def test_layer_access(self):
        self.assertEqual(self.output.get_link("Input.test_image")[0], self.input)

if __name__ == '__main__':
    unittest.main()
