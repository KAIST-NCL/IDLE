# -*- coding: utf-8 -*-

from .layer import Layer
from .inout import InOut
import re

class DLNetwork(object):
    """
    사용자의 입력을 DL-MDL로 받아서, 이를 바탕으로 DLNetwork를 구성하는 클래스
    클래스가 생성되면서 construct_network 클래스 함수를 호출해 layers를 생성하고 network를 구성
    :param parser: DL-MDL
    """

    def __init__(self, parser, framework='caffe'):
        self.parser = parser
        self.layers = {}
        self.global_store = {}
        self.argument_store = {}
        self.forward = []
        self.framework = framework
        self.construct_network()


    def construct_network(self):
        """
        DL-MDL에 각각의 layer정보를 parsing하는 resolver 클래스들을 생성해 layer들을 생성하고, input과 output정보를 이용해 layer connection

        """
        layers_def = self.parser.get_network()  # parser는 Serializer 객체이므로, get_network 함수를 호출하여 DL-MDL에서 layers 이름의 배열정보들을 반환하여 layers_def 배열에 저장
        for layer_def in layers_def:
            resolver = self.parser.new_layer_resolver(layer_def)  # Serializer 객체의 new_layer_resolver 함수를 호출해 layers_def 배열에 저장된 각각의 layer_def를 인자로 넣어 resolver 객체 생성
            self.layers[resolver.get_attr('name')] = Layer(resolver=resolver,
                                                           framework=self.framework)

        for key, layer in self.layers.items():  # key = resolver.get_attr('name') / layer = Layer(resolver=resolver)
            for input_name, input_list in layer.input.get_input_tuple():
                for output_name in input_list.keys():
                    layer.add_link(input_name, self.layers[InOut.get_split_name(output_name)[0]])
        self.traverse()

    def traverse(self):
        """
        bfs(너비 우선 검색) 알고리즘을 이용한 layer들의 operation 순차를 저장
        """
        #queue = self.get_layer_by_type('jpeg_input')

        #twkim
        queue = self.get_layer_by_input_type()

        while len(queue):
            layer = queue.pop(0)
            if layer.add_to_visit_count(-1) <= 0:
                queue.extend(layer.get_all_next_layer())
                self.forward.append(layer)
        print(self.forward)

    def compile(self, learning_option, cluster):
        for layer in self.forward:
            layer.compile_time_operate(learning_option, cluster)

    def run(self, learning_option, cluster):
        for layer in self.forward:
            layer.run_time_operate(learning_option, cluster)

    def get_all_outputs(self):
        return [layer.get_op_outputs() for layer in self.forward]

    def get_layer(self, layer_name):
        return self.layers[layer_name]

    def get_layer_by_type(self, layer_type):
        return [layer for layer in self.layers.values() if re.search(layer_type, layer.type)]

    #twkim: find input layers (current version not support the multi-input layer)
    def get_layer_by_input_type(self):
        #currently supported input types
        #twkim-test
        return [layer for layer in self.layers.values() if re.search('_input', layer.type)]