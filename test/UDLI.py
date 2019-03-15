# -*- coding: utf-8 -*-

from src.DLMDL.DLNetwork import DLNetwork
from src.Serializer.serializer import JsonSerialize
import importlib
import argparse
from src.util.signal_handler import signal_handler

class UDLI:

    def __init__(self):
        self.args = arg_parser().parse_args()
        self.handler = signal_handler()
        self.serializer = JsonSerialize()
        self.serializer.parse('%s.dlmdl' % self.getArgs('input'))
        framework = self.getArgs('framework')
        self.network = DLNetwork(self.serializer, framework)
        self.learning_option = dict(self.serializer.get_learning_option())
        self.cluster = self.serializer.get_cluster()
        self.adaptor = getattr(importlib.import_module('src.DLMDL.{framework}_adaptor'.format(framework=framework)),
                          '{framework}_adaptor'.format(framework=framework))

    def launch(self):
        # TODO: Compile / Run 시에 learning_option에서 악의적인 Conflict 발생 가능성
        if self.getArgs('run'):
            self.adaptor.compile(self)
        if self.getArgs('compile'):
            self.adaptor.run(self)

    def getArgs(self, attr):
        return getattr(self.args, attr)

    def addHandler(self, handler):
        self.handler.addHandler(handler)

def arg_parser():
    """
    argument parser : here, we add some arguments we want to use
    :return: parser
    """
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.datasets.ClusterSpec
    parser.add_argument(
        "--input",
        type=str,
        default="data/VGG", #_ModelParallel.dlmdl",
        help="input DLMDL file path"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="tf",
        help="Choose output framework - tf: tensorflow, caffe: caffe"
    )
    parser.add_argument(
        "-r",
        '--run',
        default=False,
        action='store_true',
        help="do run mode"
    )
    parser.add_argument(
        "-c",
        '--compile',
        default=False,
        action='store_true',
        help="do compile mode"
    )
    parser.add_argument(
        '--compile_out',
        type=str,
        default="output/cout",
        help="compile result file"
    )
    parser.add_argument(
        '--log_out',
        type=str,
        default="/tmp/DLMDL_log/TF",
        help="log file path. default: /tmp/DLMDL_log/TF"
    )
    parser.add_argument(
        '--parameter_out',
        type=str,
        default="output/checkpoint",
        help="checkpoint file / now unused"
    )
    return parser

if __name__ == '__main__':
    import os
    print(os.environ['PYTHONPATH'])
    UDLI().launch()