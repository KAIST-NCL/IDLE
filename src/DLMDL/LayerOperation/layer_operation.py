# -*- coding: utf-8 -*-

import abc
import inspect
import importlib
import json
import sys


class LayerOperationAdaptor:
    def __init__(self, layer):
        self.layer = layer
        current_module = sys.modules[__name__]
        module_name = 'op_{framework}_{lower_module}'.format(framework=self.layer.framework, lower_module=self.layer.type.lower())
        module = getattr(current_module, module_name, None)
        if not module:
            module = getattr(importlib.import_module('.{framework}.{module}'.format(framework=self.layer.framework, module=self.layer.type),
                package=__package__), module_name)
        self.op = module(layer)

    def run_time_operation(self, learning_option, cluster):
        print('{cls}.{func} operated!'.format(cls=self.op.__class__.__name__, func='run_time_operation'))
        self.op.run_time_operation(learning_option, cluster)

    def compile_time_operation(self, learning_option, cluster):
        print('{cls}.{func} operated!'.format(cls=self.op.__class__.__name__, func='compile_time_operation'))
        self.op.compile_time_operation(learning_option, cluster)


class LayerOperation:

    def __init__(self, layer):
        self._layer = layer
        self._input = layer.input
        self._output = layer.output
        self.outputs_list = layer.outputs_list  # get outputs attribute's values - msjeon
        self.name = layer.get_attr('name')
        self.framework = layer.framework
        attr_descriptions = json.loads(getattr(self.__class__, '_attributes'))
        self._attr = {}
        for attr in attr_descriptions:
            setattr(self, attr.get('name'), layer.get_attr(attr.get('name'), attr.get('default', None)))
            # TODO: 만약 Mandatory인데 attr이 None이면 Validation Error

    def get_attr(self, attr_name, default=None):
        return self._layer.get_attr(attr_name, default)

    def get_input(self, in_name):
        return self._layer.get_op_input(in_name)

    def set_output(self, out_name, output):
        return self._output.set_obj(out_name, output)

    def get_dimension(self, in_name):
        return self._input.get_dimension(in_name)

    def set_dimension(self, out_name, dimension):
        self._output.set_dimension(out_name, dimension)

    @abc.abstractmethod
    def compile_time_operation(self, learning_option, cluster):
        pass

    @abc.abstractmethod
    def run_time_operation(self, learning_option, cluster):
        pass
