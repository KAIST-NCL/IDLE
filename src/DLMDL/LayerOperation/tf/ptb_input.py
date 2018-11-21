from ..layer_operation import LayerOperation
import tensorflow as tf
import re


class op_tf_ptb_input(LayerOperation):
    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define input placeholder for PTB dataset
        ourputs:
            text: PTB text
            targets: one step shifted data
        """

        # get attr
        # required field
        num_steps = self.get_attr('num_steps', default=None)  # input text length at once
        if num_steps is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_steps', self.name))
        hidden_size = self.get_attr('hidden_size', default=None)  # word embedding size
        if hidden_size is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('hidden_size', self.name))
        num_class = 10000 # fixed number of word classes

        # set to learning option
        learning_option['num_steps'] = num_steps
        learning_option['num_class'] = num_class
        learning_option['hidden_size'] = hidden_size

        learning_option['classes'] = hidden_size

        # whether or not test procedure
        is_train = tf.placeholder_with_default(True, shape=())
        learning_option['is_train'] = is_train

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            # get or create embedding variable
            embedding = tf.get_variable(name='embedding',
                                        shape=[num_class, hidden_size], dtype=tf.float32)

            # input data & target placeholder
            input_data = tf.placeholder(dtype=tf.int32, shape=(None, num_steps), name='ptb_input')
            targets = tf.placeholder(dtype=tf.int32, shape=(None, num_steps), name='ptb_targets')

            # input placeholder setting to learning_option
            learning_option['data_placeholder'] = input_data
            learning_option['is_image'] = False

            # whether or not test procedure
            is_train = tf.placeholder_with_default(True, shape=())
            learning_option['is_train'] = is_train

            # word embedding for input
            input_data_ = tf.nn.embedding_lookup(embedding, input_data)

            # set output
            self.set_output('text', input_data_)
            self.set_output('targets', targets)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()

