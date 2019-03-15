from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_softmaxwithloss(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define softmax cross entropy between logits and labels
        outputs:
            output: loss output
        """
        # get input
        logits = self.get_input('logits')
        labels = self.get_input('labels')
        indim = self.get_dimension('logits')

        # get optional field
        scope = self.get_attr('scope', default=None)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        def apiConstructor():
            if learning_option.get('train_imagenet'): #TODO: tmp
                softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            else:
                softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

            loss = tf.reduce_mean(softmax_cross_entropy_with_logits)
            reg_losses = tf.losses.get_regularization_losses()
            if len(reg_losses):
                #TODO scope must contained
                loss = loss + tf.add_n(reg_losses)

            # set output
            self.set_output('output', loss)

            # set tf summary
            if scope is not None:
                tf.summary.scalar(self.name, loss, collections=[scope])
            else:
                tf.summary.scalar(self.name, loss)
        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()
