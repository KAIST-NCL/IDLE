from ..layer_operation import LayerOperation
import caffe
from caffe import layers as L

from src.DLMDL.caffe_adaptor import tempNet  #TODO: If the solver is created or changed separately in the DLMDL, review is required.

class op_caffe_jpeg_input(LayerOperation):
    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define input placeholder for JPEG input.
        """

        # get attr from learning option
        option = learning_option.get('option') #TODO: only consider training
        file_format = learning_option.get("file_format")
        data_path = learning_option.get('data_path')
        label_path = learning_option.get('label_path')
        batch_size = learning_option.get('batch_size')
        iteration = learning_option.get('iteration')

        # get attr
        # required field
        image_size = self.get_attr('image_size', default=None)
        if image_size is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('image_size', self.name))
        num_class= self.get_attr('num_class', default=None)
        if num_class is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_classes', self.name))

        # optional field
        shuffle = self.get_attr('shuffle', default=False)

        # Image Data layer setting
        image, label = L.ImageData(name=self.name,
                                   source=label_path.split('.')[0] + '_caffelist.txt',
                                   batch_size=batch_size, include=dict(phase=caffe.TRAIN), ntop=2, root_folder=data_path,
                                   new_height=image_size[1], new_width=image_size[0], shuffle=shuffle)

        test_data_path = learning_option.get('test_data_path')
        test_label_path = learning_option.get('test_label_path')
        test_batch_size = learning_option.get('test_batch_size')
        test_iteration = learning_option.get("test_iteration")
        test_interval = learning_option.get("test_interval")

        # Image Data layer for test dataset
        if test_data_path is not None:
            # Test image data layer setting
            temp_image, temp_label = L.ImageData(name=self.name,
                                                 source=test_label_path.split('.')[
                                                     0] + '_caffelist.txt',
                                                 batch_size=test_batch_size, include=dict(phase=caffe.TEST), ntop=2,
                                                 root_folder=test_data_path, new_height=image_size[0],
                                                 new_width=image_size[1])
            setattr(tempNet, str(self.name) + '.image', temp_image)
            setattr(tempNet, str(self.name) + '.label', temp_label)

        image_size = [batch_size, 3, image_size[0], image_size[1]]

        # Record the layer output information
        self.set_output('image', image)
        self.set_output('label', label)
        self.set_dimension('image', image_size)

        # required value set to learning_option
        learning_option['max_iter'] = iteration
        learning_option['test_iter'] = test_iteration
        learning_option['test_interval'] = test_interval

    def run_time_operation(self, learning_option, cluster):
        pass
