from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_multi_cells(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define sequential multi cells for input tensor
        outputs:
            output: multi cells
        """
        # get input
        # TODO: to be change 'input' to 'cell
        input_ = self.get_input('input')
        indim = self.get_dimension('input')

        # get attr
        # required field
        num_cells = self.get_attr('num_cells', default=None)
        if num_cells is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('num_cells', self.name))

        # get optional attr
        scope = self.get_attr('scope', default='default')
        # TODO: tmp
        if scope is None:
            scope = self.name

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        # construct API
        def apiConstructor():
            multi_cells = tf.contrib.rnn.MultiRNNCell([input_ for _ in range(num_cells)], state_is_tuple=True)

            # set output
            self.set_output('output', multi_cells)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()

