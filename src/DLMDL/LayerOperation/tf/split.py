from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_split(LayerOperation):
    _attributes = """[]"""  # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define split operation for input tensor
        outputs:
            output0: split output
            output1:
            output2:
            ...
        """
        # get input
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        num_split = self.get_attr('num_split', default=None)
        if num_split is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_split', self.name))
        # optional field
        axis = self.get_attr('axis', default=0)
        size_split = self.get_attr('size_split', default=None)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():

            # if size_split is specified, use num_or_size_splits argument to size_split
            # else, use num_or_size_Splits argument to num_split
            if size_split is not None:
                if len(size_split) != num_split:
                    raise Exception('[DLMDL ERROR]: number of components in size_split '
                                    'must be equal to num_split in {0} layer'.format(self.name))
                num_or_size_split = size_split
            else:
                num_or_size_split = num_split

            split = tf.split(input_, num_or_size_split, axis=axis)

            # set output
            for i in xrange(num_split):
                # get output dimension
                outdim = list(split[i].get_shape()[j].value for j in xrange(len(split[i].get_shape())))

                self.set_dimension('output{}'.format(i), outdim)
                self.set_output('output{}'.format(i), split[i])

                # set tf summary
                tf.summary.histogram(self.name, split[i])

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()