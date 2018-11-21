def get_lr_scheduler(lr_scheduler):
    """
    get caffe learning rate scheduler dictionary
    :param lr_scheduler: DLMDL learning rate scheduler declaration.
    :return: learning rate scheudler dictionary
    """
    def get_value(key, default=None):
        """
        get attributes in learning rate scheduler
        :param key: key value for parsing
        :param default: default value of attribute
        :return: value
        """
        value = lr_scheduler.get(key)
        if value is None:
            return default
        return value

    # get learning rate scheduler type
    # default: fix learning rate
    type = get_value('type', default='fix')
    if type == 'fix':
        lr_scheduler = {'lr_policy': 'fixed'}
    elif type == 'step':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        lr_scheduler =  {'lr_policy': type, 'stepsize': step, 'gamma': factor}
    elif type == 'exp':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        lr_scheduler = {'lr_policy': type, 'gamma': factor}
    elif type == 'inv':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        power = get_value('power', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('power'))
        lr_scheduler = {'lr_policy': type, 'gamma': factor, 'power': power}
    elif type == 'multi_steps':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        steplist = get_value('steplist', default=None) # WARNING: list
        if steplist is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('steplist'))
        lr_scheduler = {'lr_policy': 'multistep', 'gamma': factor, 'stepvalue': steplist}
    elif type == 'poly':
        power = get_value('power', default=None)
        if power is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('power'))
        lr_scheduler = {'type': type, 'power': power}
    elif type == 'sigmoid':
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        lr_scheduler ={'type': type, 'gamma': factor, 'stepsize': step}
    else:  # TODO: error control
        lr_scheduler = None
    return lr_scheduler
