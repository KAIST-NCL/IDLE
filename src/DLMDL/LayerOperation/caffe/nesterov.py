from ..layer_operation import LayerOperation
from lr_scheduler import get_lr_scheduler

class op_caffe_nesterov(LayerOperation):

    _attributes = """[]""" #TODO: TO BE DEPRECATED

    def compile_time_operation(self, learning_option, cluster):
        """
        define nesterov optimizer given loss and weights/biases
        """
        # get attr
        # required field
        lr = float(self.get_attr('lr', default=None))
        if lr is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('lr', self.name))
        mom = float(self.get_attr('momentum', default=None))
        if mom is None:
            raise Exception('[DLMDL ERROR]: {0} in {1} layer must be declared.'.format('momentum', self.name))

        # optional field
        lr_scheduler = self.get_attr('lr_scheduler', default={})  # default will set later
        lr_dic = get_lr_scheduler(lr_scheduler)
        opt_dic = {'type': 'Nesterov', 'base_lr': lr, 'momentum': mom}

        # setting to learning option
        learning_option['opt_dic'] = opt_dic
        learning_option['lr_sched_dic'] = lr_dic

    def run_time_operation(self, learning_option, cluster):
        pass
