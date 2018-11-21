import unittest
from src.Serializer.serializer import *


class JsonSerializerTestCase(unittest.TestCase):

    def setUp(self):
        self.serializer = JsonSerialize()
        self.serializer.parse('../data/unit_test_input.json')
        layers_def = self.serializer.get_network()
        self.resolver = []
        for layer_def in layers_def:
            self.resolver.append(self.serializer.new_layer_resolver(layer_def))

    def tearDown(self):
        self.serializer = None
        self.resolver = []

    def test_file_path(self):
        self.assertRaises(IOError, JsonSerialize().parse, '../data/abc.json')

    def test_json_attr(self):
        self.assertEqual(self.resolver[0].get_attr('type'), 'input')

    def test_json_input(self):
        self.assertEqual(self.resolver[4].get_attr('inputs')['input'][0], 'output')


class PBSerializerTestCase(unittest.TestCase):

    def test_file_path(self):
        self.assertRaises(IOError, PBSerializer().parse, '../data/abc.json')

if __name__ == '__main__':
    unittest.main()
