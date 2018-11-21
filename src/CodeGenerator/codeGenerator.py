#-*- coding: utf-8 -*-
"""
    codeGenerator.py

    Contributor : Eunjoo Yang (yejyang@kaist.ac.kr)
    Last Update : 2017.06.06

    CondeGenerator generates client code from user input using layer modules
"""
import argparse
# from __future__ import print_function # TODO --commented by msjeon

from src.DLMDL.layer import *
from src.util import codegenUtil


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
        default="../../data/DLMDL_input.json",
        help="input file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../out/clientTF.py",
        help="output file"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="tf",
        help="Choose output framework - tf : tensorflow"
    )

    return parser

class codeGenerator:

    framework = "tf"

    def find_input_layer(self, input_name):
        for Layer in self.layers:
            if input_name == Layer.name:
                return Layer

    def setTriainConfig(self, tConfig, cConfig):
        """
        set Training Configuration
        :param tConfig: training configuration - for training model
        :param cConfig: cluster configuration - for cluster spec
        :return:
        """
        # Training Configuration
        self.inputW = tConfig['image_size'][0]
        self.inputH = tConfig['image_size'][1]
        self.outputShape = tConfig['output_shape']
        self.batchSize = tConfig['batch_size']
        self.iteration = tConfig['iteration']
        self.image_size = tConfig['image_size']

        # Device Configuration
        self.NoWorker = 0
        for node in cConfig:
            for worker in node['node']['worker']:
                self.NoWorker += 1


    def generateTFCode(self):
        """
        generate Client Code
        :return:  return string
        """

        clientCode = []

        # set frameowkr as Tensorflow
        layer.framework = "TF"

        # add some import codes
        clientCode.append("import tensorflow as tf")
        clientCode.append("from layer import *\n\n")
        clientCode.append("batch_size = %d"%self.batchSize)
        clientCode.append("iteration = %d"%self.iteration)
        clientCode.append("image_size = %d"%self.image_size[0])


        # device 정보를 바탕으로 필요한 def function 개수 나눠서
        node_layers = np.zeros([self.NoWorker, 1])

        # Write Body Code
        if self.NoWorker == 1:
            # single worker code
            # generate object
            funcBody = []
            funcBody.append("\n# create layer objects\n")
            for Layer in self.layers:
                funcBody.append(Layer.createLayer())

            # add logits function definition
            clientCode = clientCode + (codegenUtil._writedef("logits", "", funcbody=funcBody))


        else:
            # multiple workers code
            for ilayer in self.layers:
                node_layers[int(ilayer.device)].append(ilayer)

            for worker_ in range(self.NoWorker):
                workerCode = []
                for Layer in node_layers[worker_]:
                    workerCode.append(Layer.createLayer())
                    workerCode.append(Layer.name + "=" + Layer.name+".TFOperation()")
                clientCode.append(codegenUtil._writedef("worker%d" % worker_, "", workerCode))


        return clientCode


if __name__ == '__main__':

    # Argument Parsing --> Input Network
    parser = arg_parser()
    args = parser.parse_args()
    # generate codeGenerator object
    codegen = codeGenerator()
    codegen.framework = args.framework

    # Open Input File and JSON Parsing


    # Generate Layer Instances
    codegen.layerParsing(input_data['layers'])

    # Set Training Configuration
    codegen.setTriainConfig(input_data['layers'][0], input_data['cluster'])

    # Generate Client Code
    if codegen.framework == "tf":
        TFCode = codegen.generateTFCode()

    # Write Client Code
    with open(args.output,"w") as clientFile:
        for codeline in TFCode:
            codeline += "\n"
            clientFile.write(codeline)

