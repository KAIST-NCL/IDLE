from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_mnist_input(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define input placeholder for MNIST input
        outputs:
            images: image data placeholder
            label: label data placeholer
        """

        # get attr
        # optional field
        num_steps = self.get_attr('num_steps', default=False)
        num_input = 28
        hidden_size = self.get_attr('hidden_size', default=False)
        num_class = 10 #fixed

        # set attribute to learning option to be used in tf_adaptor.py
        learning_option['num_steps'] = num_steps
        learning_option['num_input'] = num_input
        learning_option['hidden_size'] = hidden_size
        learning_option['is_image'] = True
        learning_option['num_class'] = 10

        # whether or not test procedure
        is_train = tf.placeholder_with_default(True, shape=())
        learning_option['is_train'] = is_train

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            # images: [num_steps, num_input]
            # label of MNIST: [10]
            if num_steps is not None: # For RNN/LSTM
                images_placeholder = tf.placeholder(tf.float32, shape=[None, num_steps, num_input])
                labels_placeholder = tf.placeholder(tf.int64 , shape=[None, num_class])
            else:
                images_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28))
                labels_placeholder = tf.placeholder(tf.int64, shape=(None, num_class))
            # get output dimension
            outdim = list(images_placeholder.get_shape()[i].value for i in xrange(len(images_placeholder.get_shape())))

            # set output
            self.set_output('image', images_placeholder)
            self.set_output('label', labels_placeholder)
            self.set_dimension('image', outdim)

            # set tf summary
            if num_steps is None:
                tf.summary.image(self.name, images_placeholder, max_outputs=10)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()



