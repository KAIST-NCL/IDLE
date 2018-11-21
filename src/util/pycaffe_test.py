import caffe
from caffe import layers as L

temp_image, temp_label = L.Data(name='tw', source='/home/ncl/caffe/examples/mnist/mnist_leveldb', batch_size=128, include=dict(phase=caffe.TRAIN), ntop=2, backend=0)
