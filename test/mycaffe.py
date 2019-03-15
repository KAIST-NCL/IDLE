import sys
sys.path.append('/Users/SeongHwanKim/tensorflow3/caffe-master/python')

import caffe
from caffe import layers as L, params as P


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    data, label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    conv1 = L.Convolution(data, name='Conv1', kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    pool1 = L.Pooling(conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    conv2 = L.Convolution(pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    pool2 = L.Pooling(conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    fc1 = L.InnerProduct(pool2, num_output=500, weight_filler=dict(type='xavier'))
    relu1 = L.ReLU(fc1, in_place=True)
    n.score = L.InnerProduct(relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, label, name='loss')

    return n.to_proto()


with open('../output/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('../output/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))