#-*- coding: utf-8 -*-

from .inout import *
from .LayerOperation.layer_operation import LayerOperationAdaptor


class Layer(object):

    def __init__(self, type=None, name=None, device=None, attr=None, inputs=None, outputs=None, framework='caffe', resolver=None):
        if resolver is not None:
            self.__init__(type=resolver.get_attr('type'), name=resolver.get_attr('name'),
                          device=resolver.get_attr('device'), attr=resolver.get_attr('attributes'),
                          inputs=resolver.get_attr('inputs'), outputs=resolver.get_attr('outputs'), framework=framework)
        else:
            self.type = type
            self.name = name
            self.device = device
            self.attr = attr

            self.input = In(self, inputs)
            self.output = Out(self, outputs)
            self.outputs_list = outputs # get outputs attribute's values - msjeon

            self.framework = framework
            self.visit_count = 0

            self.op = LayerOperationAdaptor(self)

    # for network construction
    def add_link(self, in_name, prev_layer):
        """
        layer간 연결을 하는 함수, layer에서 link를 저장하고 관리하는 layer.input 객체의 add_link 함수 호출하고, 인자로 input name과 이전 layer의 output 객체를 전달
        :param in_name: input name
        :param prev_layer: input name을 보낸 이전 layer 객체
        """
        self.input.add_link(in_name, prev_layer.output)

    def get_op_input(self, in_name):
        return self.input.get_obj(in_name)

    def set_op_output(self, output_name, output_obj):
        self.output.set_obj(self.get_full_name(output_name), output_obj)

    def get_op_output(self, output_name):
        return self.output.get_obj(self.get_full_name(output_name))

    def get_op_outputs(self):
        return [(arg_name, self.output.get_obj(arg_name)) for arg_name in self.output.get_full_args()]

    def get_attr(self, attr_name, default=None):
        if attr_name in ['type', 'name', 'device', 'framework']:
            return getattr(self, attr_name)
        if default is None:
            return self.attr.get(attr_name)
        else:
            return self.attr.get(attr_name, default)

    def run_time_operate(self, learning_option, cluster):
        self.op.run_time_operation(learning_option, cluster)

    def compile_time_operate(self, learning_option, cluster):
        self.op.compile_time_operation(learning_option, cluster)

    def get_prev_layer(self, arg_name, conn_name):
        return self.input.get_link(arg_name=arg_name, conn_name=conn_name).get_layer()

    def get_next_layer(self, arg_name):
        return [_input.get_layer() for _input in self.output.get_link(arg_name=arg_name)]

    def get_all_next_layer(self):
        layer = []
        for out_name in self.output.get_full_args():
            layer.extend(self.get_next_layer(out_name))
        return layer

    def add_to_visit_count(self, value):
        """

        :param value:
        :return:
        """
        self.visit_count += value
        return self.visit_count

    def __repr__(self):
        return "Layer ({name})".format(name=self.name)

    def get_full_name(self, arg_name):
        return '{layer_name}.{arg_name}'.format(layer_name=self.name, arg_name=arg_name)
