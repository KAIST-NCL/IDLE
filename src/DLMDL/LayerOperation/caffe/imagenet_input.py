from ..layer_operation import LayerOperation
import caffe
from caffe import layers as L

from src.DLMDL.caffe_adaptor import tempNet  #TODO: If the solver is created or changed separately in the DLMDL, review is required.

#TODO: imagenet input(data_path in learning_option) with only LMDB format
# In the future, DLMDL automatically downloads dataset and convert into LMDB foramt
class op_caffe_imagenet_input(LayerOperation):
    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):

        data_path = learning_option.get('data_path') # directory which binary files are saved
        batch_size = learning_option.get('batch_size') # train batch size

        # get attr
        # required field
        backend_idx = 1 #fixed: Imagenet with LMDB
        image_size = [batch_size, 3, 224, 224] #fixed: Imagenet dataset

        # for shmcaffe ETRI
        # learning_option["move_rate"] = learning_option.get("move_rate", 0.2)
        # learning_option["tau"] = learning_option.get("tau", 1)

        # Image Data layer setting
        image, label = L.Data(name=self.name, source=data_path, batch_size=batch_size,
                              include=dict(phase=caffe.TRAIN), ntop=2, backend=backend_idx)

        test_data_path = learning_option.get('test_data_path')
        test_batch_size = learning_option.get('test_batch_size')

        # Image Data layer for test dataset
        if test_data_path is not None:
            # Test image data layer setting
            temp_image, temp_label = L.Data(name=self.name, source=test_data_path, batch_size=test_batch_size,
                                            include=dict(phase=caffe.TEST), ntop=2, backend=backend_idx)
            setattr(tempNet, str(self.name) + '.image', temp_image)
            setattr(tempNet, str(self.name) + '.label', temp_label)

        # Record the layer output information
        self.set_output('image', image)
        self.set_output('label', label)
        self.set_dimension('image', image_size)


        """
        WARNING: delete learning_option is moved to caffe_adaptor.py
        try:
            if test_data_path is None:
                del learning_option['option']
            del learning_option['file_format']
            del learning_option['data_path']
            del learning_option['batch_size']
            del learning_option['iteration']
            del learning_option['image_size']
            learning_option['max_iter'] = iteration

            # for shmcaffe ETRI
            #del learning_option['move_rate']
            #del learning_option['tau']
        except KeyError:
            pass
        try:
            del learning_option['test_data_path']
            del learning_option['test_batch_size']
            learning_option['test_iter'] = test_iteration
            learning_option['test_interval'] = test_interval
            del learning_option['test_iteration']
        except KeyError:
            pass
        """
    def run_time_operation(self, learning_option, cluster):
        pass
