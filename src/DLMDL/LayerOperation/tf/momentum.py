from ..layer_operation import LayerOperation
import tensorflow as tf
import re
from lr_scheduler import get_lr_scheduler

class op_tf_momentum(LayerOperation):

    _attributes = """[]""" # TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        """
        define momentum optimizer given loss and weights/biases
        outputs:
            output: training op
            global_step: training global step
        """
        # get input
        loss = self.get_input('loss')

        # get attr
        # required field
        lr = float(self.get_attr('lr', default=None))
        if lr is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('lr', self.name))
        mom = float(self.get_attr('momentum', default=None))
        if mom is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('momentum', self.name))

        # optional field
        lr_scheduler = self.get_attr('lr_scheduler', default={}) # default will set later
        clip_grad = self.get_attr('clip_grad', default=None)
        scope = self.get_attr('scope', default=None)

        # get worker info: worker num, device type, device num
        device = self.get_attr('device')
        num = re.sub('[^0-9]', '', cluster.get('types')[device])
        type = cluster.get('types')[device].replace(str(num), '')

        def apiConstructor():
            # get trainable variables
            train_vars = tf.trainable_variables()
            if scope is not None:
                opt_vars = [var for var in train_vars if scope in var.name]
            else:
                opt_vars = train_vars

            lr_method = get_lr_scheduler(lr_scheduler, lr)
            global_step = tf.train.get_or_create_global_step()
            momentum = tf.train.MomentumOptimizer(lr_method, mom)
            if clip_grad is not None:
                grads = momentum.compute_gradients(loss, var_list=opt_vars)
                clipped_grads = [(tf.clip_by_value(grad, -1.0*clip_grad, 1.0*clip_grad), var) for grad, var in grads]
                train_op = momentum.apply_gradients(clipped_grads, global_step=global_step)
            else:
                train_op = momentum.minimize(loss, global_step=global_step, var_list=opt_vars)

            # set output
            self.set_output('output', train_op)
            self.set_output('global_step', global_step)

        with tf.variable_scope(self.name):
            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                    apiConstructor()
            else:
                apiConstructor()
