from ..layer_operation import LayerOperation
from lr_scheduler import get_lr_scheduler

class op_caffe_adam(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define adam optimizer given loss and weights/biases
        """
        # get attr
        # required field
        lr = float(self.get_attr('lr', default=None))
        if lr is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('lr', self.name))

        # optional field
        beta1 = float(self.get_attr('beta1', default=0.9))
        beta2 = float(self.get_attr('beta2', default=0.999))
        eps = float(self.get_attr('epsilon', default=10 ** -8))
        lr_scheduler = self.get_attr('lr_scheduler', default={})  # default will set later
        lr_dic = get_lr_scheduler(lr_scheduler)
        opt_dic = {'type': 'Adam', 'base_lr': lr, 'momentum': beta1, 'momentum2': beta2, 'delta': eps}

        # setting to learning option
        learning_option['opt_dic'] = opt_dic
        learning_option['lr_sched_dic'] = lr_dic

    def run_time_operation(self, learning_option, cluster):
        pass
