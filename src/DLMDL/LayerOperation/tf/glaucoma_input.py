from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_glaucoma_input(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define input placeholder for glaucoma input(provided by Pdxen).
        outputs:
            images: image data placeholder
            label: label data placeholder
        """

        # get attr
        # optional field
        shuffle = self.get_attr('shuffle', default=False)

        # set attribute to learning option to be used in tf_adaptor.py
        learning_option['shuffle'] = shuffle
        learning_option['is_image'] = True

        # whether or not test procedure
        is_train = tf.placeholder_with_default(True, shape=())
        learning_option['is_train'] = is_train

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            # randomly cropped images: [224, 224, 3]
            # label of diagnosis: [2]
            images_placeholder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='glaucoma_aug_image')
            labels_placeholder = tf.placeholder(tf.int64, shape=(None, 2), name='galucoma_aug_label')

            # get output dimension
            outdim = list(images_placeholder.get_shape()[i].value for i in xrange(len(images_placeholder.get_shape())))

            # set output
            self.set_output('image', images_placeholder)
            self.set_output('label', labels_placeholder)
            self.set_dimension('image', outdim)

            # set tf summary
            tf.summary.image(self.name, images_placeholder, max_outputs=10)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()



