from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_imagenet_input(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        get input tensor for imagenet image.
        outputs:
            images: image data tensor
            label: label data tensor
        """

        # whether or not test procedure
        is_train = tf.placeholder_with_default(True, shape=())
        learning_option['is_train'] = is_train

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            # CIFAR-10 images: [224, 224, 3]
            # label: [1000]
            def train_in():
                x, y = learning_option.get('train_imagenet')
                return x, y
            def test_in():
                x, y = learning_option.get('test_imagenet')
                return x, y

            images, labels = tf.cond(is_train, train_in, test_in)
            # get output dimension
            outdim = list(images.get_shape()[i].value for i in xrange(len(images.get_shape())))

            # set output
            self.set_output('image', images)
            self.set_output('label', labels)
            self.set_dimension('image', outdim)

            # set tf summary
            tf.summary.image(self.name, images, max_outputs=10)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()



