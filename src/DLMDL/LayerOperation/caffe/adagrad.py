from ..layer_operation import LayerOperation
from lr_scheduler import get_lr_scheduler

class op_caffe_adagrad(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define adagrad optimizer given loss and weights/biases
        refer: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, Journal of Machine Learning Research 2011
        """
        # get attr
        # required field
        lr = float(self.get_attr('lr', default=None))
        if lr is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('lr', self.name))

        # optional field
        lr_scheduler = self.get_attr('lr_scheduler', default={})  # default will set later
        lr_dic = get_lr_scheduler(lr_scheduler)
        opt_dic={'type': 'AdaGrad', 'base_lr': lr}

        # setting to learning option
        learning_option['opt_dic'] = opt_dic
        learning_option['lr_sched_dic'] = lr_dic

    def run_time_operation(self, learning_option, cluster):
        pass
