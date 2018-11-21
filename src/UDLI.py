# -*- coding: utf-8 -*-

from DLMDL.DLNetwork import DLNetwork
from Serializer.serializer import JsonSerialize
import importlib
import argparse
from util.signal_handler import signal_handler
import os
import sys
import datetime


class UDLI:

    def __init__(self):
        self.args = UDLI.arg_parser().parse_args()
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
        self.adaptor.compile(self)
        if self.getArgs('run'):
            self.adaptor.run(self)

    def getArgs(self, attr):
        return getattr(self.args, attr)

    def addHandler(self, handler):
        self.handler.addHandler(handler)

    @staticmethod
    def getCWD():
        return os.getcwd()

    @staticmethod
    def getFullPath(base_filename, filename_suffix=None):
        ext = (("." + filename_suffix) if filename_suffix else "")
        if base_filename[0] == '/':
            return base_filename + ext
        else:
            return os.path.join(UDLI.getCWD(), base_filename + ext)

    @staticmethod
    def script_name():
        return UDLI.getFullPath(sys.argv[0])

    @staticmethod
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
            '--compile_out',
            type=str,
            default="output/compile/cout_"+datetime.datetime.now().strftime('%y%m%d_%H%M%S'),
            help="compile result file"
        )
        parser.add_argument(
            '--log_out',
            type=str,
            default="output/log",
            help="log file path."
        )
        parser.add_argument(
            '--parameter_out',
            type=str,
            default="output/checkpoint",
            help="checkpoint file prefix"
        )
        return parser

if __name__ == '__main__':
    UDLI().launch()