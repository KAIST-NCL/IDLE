import json
import abc
from collections import OrderedDict
from src.DLMDL.cluster_manager import cluster_manager

class Serializer(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def parse(self, file_path):
        pass

    @abc.abstractmethod
    def get_network(self):
        pass

    @abc.abstractmethod
    def new_layer_resolver(self, layer_def):
        pass

class JsonSerialize(Serializer):

    obj_def = None

    def parse(self, file_path):
        with open(file_path) as f:
            self.obj_def = json.load(f, object_pairs_hook=OrderedDict)
        self.cluster_manager = cluster_manager(self.obj_def['cluster'])
        self.cluster_manager.match_index(self.obj_def['layers'])

    def get_network(self):
        return self.obj_def['layers']

    def get_cluster(self):
        return  self.cluster_manager

    def get_learning_option(self):
        return self.obj_def['learning_option']

    def new_layer_resolver(self, layer_def):
        return JsonLayerResolver(layer_def)

class PBSerializer(Serializer):

    def parse(self, file_path):
        pass

    def new_layer_resolver(self, layer_def):
        return PBLayerResolver(layer_def)


class LayerResolver:

    __metaclass__ = abc.ABCMeta
    layer_def = None

    def __init__(self, layer_def):
        self.layer_def = layer_def


class JsonLayerResolver(LayerResolver):

    def get_attr(self, attr_name, default=None):
        try:
            return self.layer_def[attr_name]
        except:
            return default


class PBLayerResolver(LayerResolver):

    def get_attr(self, attr_name):
        pass