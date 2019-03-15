from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_sequenceloss(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        # get input
        logits = self.get_input('logits')
        targets_ = self.get_input('targets')
        indim = self.get_dimension('logits')

        '''
        TODO: TF v1.1 is not support function return in tf.cond with no name attribute.
        is_train = learning_option.get('is_train')
        bs = tf.cond(is_train, lambda: learning_option.get('batch_size'),
                     lambda: learning_option.get('test_batch_size'))
        '''
        # So we assume learning_option.get('batch_size') = learning_option.get('test_batch_size')
        # WARNING: to be changed to previous operations
        bs = learning_option.get('batch_size')

        # get attr
        # optional field
        weights = float(self.get_attr('weights', default=1.0))
        scope = self.get_attr('scope', default=None)
        # TODO: tmp
        if scope is None:
            scope = self.name

        num_steps = learning_option.get('num_steps')
        num_class = learning_option.get('num_class')

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')


        # construct API
        def apiConstructor():

            logits_ = tf.reshape(logits, [bs, num_steps, num_class]) #twkim-test
            weights_ = tf.constant(weights, dtype=tf.float32, shape=(bs, num_steps))

            loss = tf.contrib.seq2seq.sequence_loss(logits_,
                                                    targets_,
                                                    weights_,
                                                    average_across_timesteps=False,
                                                    average_across_batch=True)

            loss_ = tf.reduce_sum(loss)
            reg_losses = tf.losses.get_regularization_losses()
            if len(reg_losses):
                # TODO scope must contained
                loss_ = loss_ + tf.add_n(reg_losses)

            # set output
            self.set_output('output', loss_)

            # set tf summary
            if scope is not None:
                tf.summary.scalar(self.name, loss_, collections=[scope])
            else:
                tf.summary.scalar(self.name, loss_)

        with tf.variable_scope(self.name):
            # single node, model parallelism: explicit worker mapping
            # data parallelism: equally duplicate model
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()
