# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/SeongHwanKim/tensorflow3/caffe-master/python')
from caffe.proto import caffe_pb2
import numpy as np
import json
import copy

global mdlnet
global inlayername

# calculate output dimension size for all layer
def layerDim(mdlnet, inname, indim, sizelist):
    tempdim = copy.deepcopy(indim)
    for layer in mdlnet["layers"]:
        if "name" in layer and "input" in layer["inputs"]:
            if layer["inputs"]["input"][0].split(".")[0] == inname: #inname레이어의 다음 레이어의 경우
                if layer["type"] == "conv": #conv의 경우
                    # get stride value
                    if "stride" in layer["attributes"]:
                        temp_stride = int(layer["attributes"]["stride"])
                    else:
                        temp_stride = 1

                    # get kernel size value
                    temp_ksize = [layer["attributes"]["filter"][0], layer["attributes"]["filter"][1]]

                    # calculate dimension w.r.t padding type
                    if "padding" in layer["attributes"]:
                        if layer["attributes"]["padding"].upper() == "SAME":
                            tempdim[0] = np.ceil(float(tempdim[0]) / float(temp_stride))
                            tempdim[1] = np.ceil(float(tempdim[1]) / float(temp_stride))

                        elif layer["attributes"]["padding"].upper() == "VALID":
                            tempdim[0] = np.ceil(float(tempdim[0] - temp_ksize[0] + 1) / float(temp_stride))
                            tempdim[1] = np.ceil(float(tempdim[1] - temp_ksize[1] + 1) / float(temp_stride))

                    else:
                        tempdim[0] = np.ceil(float(tempdim[0]) / float(temp_stride))
                        tempdim[1] = np.ceil(float(tempdim[1]) / float(temp_stride))

                    # recursive
                    sizelist.append([layer["name"], copy.deepcopy(tempdim)])
                    layerDim(mdlnet, layer["name"], copy.deepcopy(tempdim), sizelist)

                elif layer["type"] == "max_pool" or layer["type"] == "avg_pool": #max_pool, avg_pool의 경우
                    # get stride value
                    if "stride" in layer["attributes"]:
                        temp_stride = int(layer["attributes"]["stride"])
                    else:
                        temp_stride = 2

                    # get kernel size value
                    if "ksize" in layer["attributes"]:
                        temp_ksize = int(layer["attributes"]["ksize"])
                    else:
                        temp_ksize = 2

                    # calculate dimension w.r.t padding type
                    if "padding" in layer["attributes"]:
                        if layer["attributes"]["padding"].upper() == "SAME":
                            tempdim[0] = np.ceil(float(tempdim[0]) / float(temp_stride))
                            tempdim[1] = np.ceil(float(tempdim[1]) / float(temp_stride))

                        elif layer["attributes"]["padding"].upper() == "VALID":
                            tempdim[0] = np.ceil(float(tempdim[0] - temp_ksize + 1) / float(temp_stride))
                            tempdim[1] = np.ceil(float(tempdim[1] - temp_ksize + 1) / float(temp_stride))

                    else:
                        tempdim[0] = np.ceil(float(tempdim[0]) / float(temp_stride))
                        tempdim[1] = np.ceil(float(tempdim[1]) / float(temp_stride))

                    # recursive
                    sizelist.append([layer["name"], copy.deepcopy(tempdim)])
                    layerDim(mdlnet, layer["name"], copy.deepcopy(tempdim), sizelist)


# read the dl-mdl model
def readDLMDL(path):
    json_data=open(path).read()
    mdlnet = json.loads(json_data)
    return mdlnet

# convert dropout attr
def dropoutConvert(net, alayer, lname, ratio):
    aalayer = net.layer.add()
    aalayer.name = lname + "_dropout"	# name
    aalayer.type = "Dropout"			# type
    aalayer.bottom.extend([lname])		# bottom    # TODO:Error?
    aalayer.top.extend([lname])			# top       # TODO:Error?
    aalayer.dropout_param.dropout_ratio = ratio # dropout ratio

# convert activation attr
def activationConvert(net, alayer, lname, actmode):
    if actmode == "relu":
        aalayer = net.layer.add()
        aalayer.name = lname + "_relu"	# name
        aalayer.type = "ReLU"			# type
        aalayer.bottom.extend([lname])	# bottom    # TODO:Error?
        aalayer.top.extend([lname])		# top       # TODO:Error?
    # else activation?????????????????????

# convert bias attr
def biasConvert(alayer, initmode, ltype):
    # check layer type
    if ltype == "conv":
        tempo = alayer.convolution_param
    elif ltype == "fc":
        tempo = alayer.inner_product_param

    # set weight filler w.r.t selected mode
    if initmode == "random_normal":
        tempo.weight_filler.type = "gaussian"
        tempo.weight_filler.mean = 0
        tempo.weight_filler.std = 1
    elif initmode == "zeroes":
        tempo.weight_filler.type = "constant"

# convert weight attr
def weightConvert(alayer, initmode, ltype):
    # check layer type
    if ltype == "conv":
        tempo = alayer.convolution_param
    elif ltype == "fc":
        tempo = alayer.inner_product_param

    # set bias filler w.r.t selected mode
    if initmode == "random_normal":
        tempo.bias_filler.type = "gaussian"
        tempo.bias_filler.mean = 0
        tempo.bias_filler.std = 1
    elif initmode == "zeroes":
        tempo.bias_filler.type = "constant"

# convert padding attr
def paddingConvert(layer, alayer, temp_stride, indim):
    # get kernel size
    tempksize = [0,0]
    if layer["type"] == "conv":
        tempksize[0] = layer["attributes"]["filter"][0]
        tempksize[1] = layer["attributes"]["filter"][1]
    elif layer["type"] == "max_pool" or layer["type"] == "avg_pool":
        tempksize[0] = int(layer["attributes"]["ksize"])
        tempksize[1] = int(layer["attributes"]["ksize"])

    # calculate padding
    if "padding" in layer["attributes"]:
        if layer["attributes"]["padding"].upper() == "VALID":
            pad_h = 0
            pad_w = 0

        elif layer["attributes"]["padding"].upper() == "SAME":
            # calculate padding value for SAME
            out_height = np.ceil(indim[0] / temp_stride)
            out_width  = np.ceil(indim[1] / temp_stride)

            pad_along_height = ((out_height - 1) * temp_stride + tempksize[0] - indim[0])
            pad_along_width = ((out_width - 1) * temp_stride + tempksize[1] - indim[1])
            pad_top = pad_along_height / 2
            pad_left = pad_along_width / 2

            pad_h = int(pad_top)
            pad_w = int(pad_left)

    else: # default to SAME
        # calculate padding value for SAME
        out_height = np.ceil(indim[0] / temp_stride);
        out_width  = np.ceil(indim[1] / temp_stride);

        pad_along_height = ((out_height - 1) * temp_stride + tempksize[0] - indim[0]);
        pad_along_width = ((out_width - 1) * temp_stride + tempksize[1] - indim[1]);
        pad_top = pad_along_height / 2;
        pad_left = pad_along_width / 2;

        pad_h = int(pad_top)
        pad_w = int(pad_left)

    return [pad_h, pad_w]

# convert input layer
def inputConvert(layer, net, sol):
    # add layer to caffe net
    alayer = net.layer.add()

    # layer checkpoint setting
    alayer.name = layer["name"]		# name
    alayer.type = "Data"			# type
    alayer.top.extend(["data"])		# top # need to be considered????????????
    alayer.top.extend(["label"])	# top # need to be considered????????????

    # include{ phase }
    if "option" in layer["attributes"]:
        temp = caffe_pb2.NetStateRule()
        if layer["attributes"]["option"].lower() == "datasets":
            temp.phase = 0
        elif layer["attributes"]["option"].lower() == "test":
            temp.phase = 1
        alayer.include.extend([temp])

    # data_param{}
    #    source
    if "train_data" in layer["attributes"]:
        alayer.data_param.source = layer["attributes"]["train_data"]
    #    batch_size
    if "batch_size" in layer["attributes"]:
        alayer.data_param.batch_size = layer["attributes"]["batch_size"]

    # solver checkpoint setting
    # solver_mode
    if "mode" in layer["attributes"]:
        if layer["attributes"]["mode"].lower() == "cpu":
            sol.solver_mode = 0
        elif layer["attributes"]["mode"].lower() == "gpu":
            sol.solver_mode = 1
    else:
        sol.solver_mode = 1
    # max_iter
    if "iteration" in layer["attributes"]:
        sol.max_iter = layer["attributes"]["iteration"]

    # default value!!!!!!!!
    alayer.data_param.backend = 0

    ## image_size ???????????????????
    ## test_data?????

# convert conv layer
def convConvert(layer, net, sizelist):
    # add layer to caffe net
    alayer = net.layer.add()

    # get input dimension
    for item in sizelist:
        print(item)
        print(layer["inputs"]["input"][0].split(".")[0])
        if item[0] == layer["inputs"]["input"][0].split(".")[0]:
            indim = item[1] # [height, width]

    # layer checkpoint setting
    alayer.name = layer["name"]	# name
    alayer.type = "Convolution"	# type

    # bottom, top, kernel_h/w, num_output
    if layer["inputs"]["input"][0].split(".")[0] == inlayername:
        alayer.bottom.extend(["data"])	# need custom handling???????????????????
    else:
        alayer.bottom.extend([layer["inputs"]["input"][0].split(".")[0]])
    alayer.top.extend([layer["name"]])
    alayer.convolution_param.kernel_h = layer["attributes"]["filter"][0]
    alayer.convolution_param.kernel_w = layer["attributes"]["filter"][1]
    alayer.convolution_param.num_output = layer["attributes"]["filter"][3]

    # stride
    if "stride" in layer["attributes"]:
        alayer.convolution_param.stride.extend([int(layer["attributes"]["stride"])])
        temp_stride = int(layer["attributes"]["stride"])
    else:
        temp_stride = 1

    # pad_h/w
    temp_pad = paddingConvert(layer, alayer, temp_stride, indim)
    alayer.convolution_param.pad_h = temp_pad[0]
    alayer.convolution_param.pad_w = temp_pad[1]

    # weight_filler
    if "weight" in layer["attributes"]:
        weightConvert(alayer, layer["attributes"]["weight"].lower(), "conv")
    else:
        weightConvert(alayer, "random_normal", "conv")

    # bias_filler
    if "bias" in layer["attributes"]:
        biasConvert(alayer, layer["attributes"]["bias"].lower(), "conv")
    else:
        biasConvert(alayer, "zeroes", "conv")

    # activation
    if "activation" in layer["attributes"]:
        activationConvert(net, alayer, layer["name"], layer["attributes"]["activation"])

    # dropout
    if "dropout" in layer["attributes"]:
        activationConvert(net, alayer, layer["name"], float(layer["attributes"]["dropout"]))


# convert pooling layer
def poolConvert(layer, net, sizelist):
    # add layer to caffe net
    alayer = net.layer.add()

    # get input dimension
    for item in sizelist:
        print(item)
        print(layer["inputs"]["input"][0].split(".")[0])
        if item[0] == layer["inputs"]["input"][0].split(".")[0]:
            indim = item[1] # [height, width]

    # layer checkpoint setting
    alayer.name = layer["name"]	# name
    alayer.type = "Pooling"		# type

    # bottom
    if layer["inputs"]["input"][0].split(".")[0] == inlayername:
        alayer.bottom.extend(["data"])	# need custom handling???????????????????
    else:
        alayer.bottom.extend([layer["inputs"]["input"][0].split(".")[0]])
    # top
    alayer.top.extend([layer["name"]])

    # pool type
    if layer["type"] == "max_pool":
        alayer.pooling_param.pool = 0
    elif layer["type"] == "avg_pool":
        alayer.pooling_param.pool = 1

    # stride
    if "stride" in layer["attributes"]:
        alayer.pooling_param.stride = int(layer["attributes"]["stride"])
        temp_stride = int(layer["attributes"]["stride"])
    else:
        alayer.pooling_param.stride = 2
        temp_stride = 2

    # kernel_size
    if "ksize" in layer["attributes"]:
        alayer.pooling_param.kernel_size = int(layer["attributes"]["ksize"])
    else:
        alayer.pooling_param.kernel_size = 2

    # pad_h/w
    temp_pad = paddingConvert(layer, alayer, temp_stride, indim)
    alayer.pooling_param.pad_h = temp_pad[0]
    alayer.pooling_param.pad_w = temp_pad[1]

    # activation
    if "activation" in layer["attributes"]:
        activationConvert(net, alayer, layer["name"], layer["attributes"]["activation"])

    # dropout
    if "dropout" in layer["attributes"]:
        activationConvert(net, alayer, layer["name"], float(layer["attributes"]["dropout"])) # TODO:Error?

# convert fc layer
def fcConvert(layer, net):
    # add layer to caffe net
    alayer = net.layer.add()

    # layer checkpoint setting
    alayer.name = layer["name"]		# name
    alayer.type = "InnerProduct"	# type

    # bottom
    if layer["inputs"]["input"][0].split(".")[0] == inlayername:
        alayer.bottom.extend(["data"])	# need custom handling???????????????????
    else:
        alayer.bottom.extend([layer["inputs"]["input"][0].split(".")[0]])
    # top
    alayer.top.extend([layer["name"]])

    # num_output
    if "output_shape" in layer["attributes"]:
        alayer.inner_product_param.num_output = int(layer["attributes"]["output_shape"])

    # weight_filler
    if "weight" in layer["attributes"]:
        weightConvert(alayer, layer["attributes"]["weight"].lower(), "fc")
    else:
        weightConvert(alayer, "random_normal", "fc")

    # bias_filler
    if "bias" in layer["attributes"]:
        biasConvert(alayer, layer["attributes"]["bias"].lower(), "fc")
    else:
        biasConvert(alayer, "zeroes", "fc")

    # activation
    if "activation" in layer["attributes"]:
        activationConvert(net, alayer, layer["name"], layer["attributes"]["activation"])

# convert xentropy loss layer
def xentropylossConvert(layer, net):
    # add layer to caffe net
    alayer = net.layer.add()

    # layer checkpoint setting
    alayer.name = layer["name"]
    alayer.type = "SoftmaxWithLoss"
    alayer.bottom.extend([layer["inputs"]["logits"][0].split(".")[0]])
    alayer.bottom.extend(["label"]) # need custom handling????????????????
    alayer.top.extend([layer["name"]])

# convert accuracy layer
def accuracyConvert(layer, net):
    # add layer to caffe net
    alayer = net.layer.add()

    # layer checkpoint setting
    alayer.name = layer["name"]
    alayer.type = "Accuracy"
    alayer.bottom.extend([layer["inputs"]["logits"][0].split(".")[0]])
    alayer.bottom.extend(["label"]) # need custom handling????????????????
    alayer.top.extend([layer["name"]])

# convert optimizer checkpoint
def optimizerConvert(layer, sol):
    # default values
    sol.net = "/home/ncl/caffe/examples/msjeon/dlmdl2caffe/dlmdl2caffe.prototxt"
    sol.lr_policy = "fixed"
    sol.display = 50

    # specified values
    sol.base_lr = float(layer["attributes"]["learning_rate"])

    if "beta1" in layer["attributes"]:
        sol.momentum = float(layer["attributes"]["beta1"])

    if "beta2" in layer["attributes"]:
        sol.momentum2 = float(layer["attributes"]["beta2"])

    ### epsilon, train_iteration(input layer), test_iteration, test_interval????????????

# convert dl-mdl => caffe for each layer
def netConvertMdl2caffe(mdlnet, net, sol, sizelist):
    for layer in mdlnet["layers"]:
        # print layer

        # convert in each type of layer
        if layer["type"] == "input":
            inputConvert(layer, net, sol)

        elif layer["type"] == "conv":
            convConvert(layer, net, sizelist)

        elif layer["type"] == "max_pool":
            poolConvert(layer, net, sizelist)

        elif layer["type"] == "avg_pool":
            poolConvert(layer, net, sizelist)

        elif layer["type"] == "fc":
            fcConvert(layer, net)

        elif layer["type"] == "loss":
            xentropylossConvert(layer, net)

        elif layer["type"] == "accuracy":
            accuracyConvert(layer, net)

        elif layer["type"] == "adamoptimizer":
            optimizerConvert(layer, sol)


"""
main
"""
# create caffe pb2 net, solver
net = caffe_pb2.NetParameter()
net.name = 'dlmdl2caffe'
sol = caffe_pb2.SolverParameter()

# read dl-mdl file
mdlnet = readDLMDL("../data/VGG.dlmdl")

# get dimension of all layer
sizelist = []
for layer in mdlnet["layers"]:
    if layer["type"] == "input":
        inlayername = copy.deepcopy(layer["name"])
        sizelist.append([layer["name"], layer["attributes"]["image_size"]])
        layerDim(mdlnet, layer["name"], layer["attributes"]["image_size"], sizelist)

# convert dl-mdl => caffe
netConvertMdl2caffe(mdlnet, net, sol, sizelist)

print(str(net))
print(str(sol))
# print str(mdlnet)
# print sizelist

# write to file
outFn = '../output/dlmdl2caffe.prototxt'
print('writing', outFn)
with open(outFn, 'w') as f:
    f.write(str(net))

outFn2 = '../output/sol.prototxt'
print('writing', outFn2)
with open(outFn2, 'w') as f2:
  f2.write(str(sol))
