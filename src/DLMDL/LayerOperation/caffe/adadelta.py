from ..layer_operation import LayerOperation
from lr_scheduler import get_lr_scheduler

class op_caffe_adadelta(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define adadelta optimizer given loss and weights/biases
        refer: ADADELTA: An Adaptive Learning Rate Method
        """
        # get attr
        # required field
        lr = float(self.get_attr('lr', default=None))
        if lr is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('lr', self.name))

        # optional field
        rho = float(self.get_attr('rho', default=0.95))
        eps = float(self.get_attr('epsilon', default=10 ** -8))
        lr_scheduler = self.get_attr('lr_scheduler', default={})  # default will set later
        lr_dic = get_lr_scheduler(lr_scheduler)
        opt_dic = {'type': 'AdaDelta', 'base_lr': lr, 'momentum': rho, 'delta': eps}

        # setting to learning option
        learning_option['opt_dic'] = opt_dic
        learning_option['lr_sched_dic'] = lr_dic

    def run_time_operation(self, learning_option, cluster):
        pass
