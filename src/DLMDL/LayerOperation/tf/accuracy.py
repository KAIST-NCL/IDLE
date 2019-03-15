from ..layer_operation import LayerOperation
import tensorflow as tf
import re

class op_tf_accuracy(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define accuracy based on logits and labels
        outputs:
            output: top-k accuracy
        """
        # get input
        logits = self.get_input('logits')
        label = self.get_input('labels')
        indim = self.get_dimension('logits')

        # get attr
        # optional field
        topk = self.get_attr('topk', default=1)
        scope = self.get_attr('scope', default=None)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        def apiConstructor():
            if learning_option.get('train_imagenet'):  # TODO: tmp
                truth = label
            else:
                truth = tf.argmax(label, axis=1)
            #truth = tf.argmax(label, axis=1)
            pred = tf.nn.softmax(logits=logits)
            predictions = tf.nn.in_top_k(pred, truth, topk)
            accuracy = tf.reduce_mean(tf.to_float(predictions))

            # set output
            self.set_output('output', accuracy)

            # set tf summary
            if scope is not None:
                tf.summary.scalar(self.name, accuracy, collections=[scope])
            else:
                tf.summary.scalar(self.name, accuracy)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()


