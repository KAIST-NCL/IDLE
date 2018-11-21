from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_bodydata_input(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define input placeholder for body data input from SNU.
        It consists of two types of data; 1) body image, 2) body status data
        outputs:
            image: image data
            bt_label: BT label for image
            bs_label: BS label for image
            numeric: status data
            numeric_label: status label
        """

        # get attr
        # optional field
        shuffle = self.get_attr('shuffle', default=False)

        # whether or not test procedure
        is_train = tf.placeholder_with_default(True, shape=())
        learning_option['is_train'] = is_train

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            images_placeholder = tf.placeholder(tf.float32, shape=(None, 64, 64, 2))
            bt_label_placeholder = tf.placeholder(tf.int32, shape=(None, 3))
            bs_label_placeholder = tf.placeholder(tf.int32, shape=(None, 11))

            numeric_placeholder = tf.placeholder(tf.float32, shape=(None, 14))
            numeric_targets_placeholder = tf.placeholder(tf.float32, shape=(None, 9))

            # get output dimension
            outdim_images = list(images_placeholder.get_shape()[i].value for i in xrange(len(images_placeholder.get_shape())))
            outdim_numeric = list(numeric_placeholder.get_shape()[i].value for i in xrange(len(numeric_placeholder.get_shape())))

            # set output
            self.set_output('image', images_placeholder)
            self.set_output('bt_label', bt_label_placeholder)
            self.set_output('bs_label', bs_label_placeholder)
            self.set_output('numeric', numeric_placeholder)
            self.set_output('numeric_label', numeric_targets_placeholder)

            self.set_dimension('image', outdim_images)
            self.set_dimension('numeric', outdim_numeric)

            # set tf summary
            #tf.summary.image(self.name, images_placeholder, max_outputs=10)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()



