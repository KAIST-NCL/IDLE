import tensorflow as tf

def get_lr_scheduler(lr_scheduler, lr):
    """
    get learning rate scheduler method
    :param lr_scheduler: DLMDL learning rate scheduler declaration.
    :return: learning rate scheudler method or fixed learning rate
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
    global_step = tf.train.get_or_create_global_step()
    if type == 'fix':
        lr_scheduler = lr
    elif type == 'step':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        lr_scheduler =  tf.train.exponential_decay(lr, global_step, step, factor, staircase=True)
    elif type == 'exp':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        lr_scheduler = tf.train.exponential_decay(lr, global_step, step, factor, staircase=False)
    elif type == 'inv':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        staircase = get_value('staircase', default=False)
        lr_scheduler = tf.train.inverse_time_decay(lr, global_step, step, factor, staircase=staircase)
    elif type == 'natural_exp':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        staircase = get_value('staircase', default=False)
        lr_scheduler = tf.train.natural_exp_decay(lr, global_step, step, factor, staircase=staircase)
    elif type == 'multi_steps':
        factor = get_value('factor', default=None)
        if factor is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('factor'))
        steplist = get_value('steplist', default=None)
        if steplist is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('steplist'))
        valuelist = []
        for idx, _ in enumerate(steplist):
            valuelist.append(factor, (idx+1))
        lr_scheduler = tf.train.piecewise_constant(global_step, steplist, valuelist)
    elif type == 'multi_values':
        valuelist = get_value('valuelist', default=None)
        if valuelist is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('valuelist'))
        steplist = get_value('steplist', default=None)
        if steplist is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('steplist'))
        lr_scheduler = tf.train.piecewise_constant(global_step, steplist, valuelist)
    elif type == 'poly':
        power = get_value('power', default=None)
        if power is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('power'))
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        end_lr = get_value('end_lr', default=0.0001)
        cycle = get_value('cycle', default=False)
        lr_scheduler = tf.train.polynomial_decay(lr, global_step, step, end_learning_rate=end_lr,
                                                 power=power, cycle=cycle)
    elif type == 'cosine':
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        alpha = float(get_value('alpha', default=0.0))
        lr_scheduler = tf.train.cosine_decay(lr, global_step, step, alpha=alpha)
    elif type == 'linear_cosine':
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        num_periods = float(get_value('num_periods', default=0.5))
        alpha = float(get_value('alpha', default=0.0))
        beta = float(get_value('beta', default=0.001))
        lr_scheduler = tf.train.linear_cosine_decay(lr, global_step, step, num_periods=num_periods,
                                                    alpha=alpha, beta=beta)
    elif type == 'noisy_linear_cosine':
        step = get_value('step', default=None)
        if step is None:
            raise Exception('[DLMDL ERROR]: {0} in learning rate scheduler must be declared.'.format('step'))
        initial_variance = float(get_value('initial_variance', default=1.0))
        variance_decay = float(get_value('variance_decay', default=0.55))
        num_periods = float(get_value('num_periods', default=0.5))
        alpha = float(get_value('alpha', default=0.0))
        beta = float(get_value('beta', default=0.001))
        lr_scheduler = tf.train.noisy_linear_cosine_decay(lr, global_step, step, initial_variance=initial_variance,
                                                          variance_decay=variance_decay, num_periods=num_periods,
                                                          alpha=alpha, beta=beta)
    else:  # TODO: error control
        lr_scheduler = None
    return lr_scheduler