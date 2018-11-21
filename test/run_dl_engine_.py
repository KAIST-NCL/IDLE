from src.DLMDL.DLNetwork import DLNetwork
from src.Serializer.serializer import JsonSerialize
import importlib
import argparse

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
        default="data/VGG.dlmdl",
        help="input file"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="tf",
        help="Choose output framework - tf: tensorflow, caffe: caffe"
    )

    return parser

if __name__ == '__main__':
    args = arg_parser().parse_args()
    framework = args.framework
    serializer = JsonSerialize()
    serializer.parse(args.input)
    network = DLNetwork(serializer, framework)
    adaptor = getattr(importlib.import_module('src.DLMDL.{framework}_adaptor'.format(framework=framework)),
                      '{framework}_adaptor'.format(framework=framework))
    adaptor.compile(network)
    adaptor.run(network)
