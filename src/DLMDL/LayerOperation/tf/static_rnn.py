from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_static_rnn(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define recurrent neural network based on input cells
        outputs:
            output: recurrent network
        """
        # get input
        # TODO: to be change "inputs" : ['input'] to "inputs": ['input', 'cell']
        cells_ = self.get_input('input')
        input_ = learning_option.get('static_rnn_input')
        #input_ = self.get_input('input')
        #indim = self.get_dimension('input')

        # get attr
        # optional field
        init = self.get_attr('initial_state', default=None)
        length = self.get_attr('length', default=None)
        scope = self.get_attr('scope', default='default')
        # TODO: tmp
        if scope is None:
            scope = self.name

        num_steps = learning_option.get('num_steps')
        is_train = learning_option.get('is_train')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            batch_size = tf.cond(is_train, lambda: tf.constant(learning_option.get('batch_size'), dtype=tf.int32),
                                 lambda: tf.constant(learning_option.get('test_batch_size'), dtype=tf.int32))
            if init == 'ZERO': # WARNING: only support zero initial state in this version
                initial_state = cells_.zero_state(batch_size, tf.float32)
            else:
                initial_state = None
            learning_option['initial_state'] = initial_state

            input_unstack = tf.unstack(input_, num=num_steps, axis=1)
            output, state = tf.contrib.rnn.static_rnn(cells_, input_unstack, initial_state=initial_state,
                                                      dtype=tf.float32, sequence_length=length)

            # set output
            self.set_output('output', output)
            self.set_output('state', state)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()

