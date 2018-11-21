def get_initializer(initializer, is_bias=False):
    """
    get caffe initializer method.
    :param initializer: DLMDL weight/bias initializer declaration. Dictionary
    :param is_bias: whether bias or not
    :return: weight/bias initializer dictinary
    """

    def get_value(key, default=None):
        """
        get attributes in initializer
        :param key: key value for parsing
        :param default: default value of attribute
        :return: value
        """
        value = initializer.get(key)
        if value is None:
            return default
        return value

    # get initializer type
    # default: weights - 'normal', bias - 'constant'
    type = get_value('type', default='constant' if is_bias else 'normal')

    # find initializer
    if type == 'constant': # constant value initializer
        value = get_value('value', default=None)
        if value is None:
            raise Exception('[DLMDL ERROR]: {0} in uniform initializer must be declared.'.format('value'))
        init = {'type': type, 'value': value}
    elif type == 'uniform': # uniform random initializer
        max = get_value('max', default=None)
        if max is None:
            raise Exception('[DLMDL ERROR]: {0} in uniform initializer must be declared.'.format('max'))
        min = get_value('min', default=0.0)
        init = {'type': type, 'min': min, 'max': max}
    elif type == 'normal': # gaussian initializer
        mean = get_value('mean', default=0.0)
        std = get_value('std', default=0.01)
        init = {'type': 'gaussian', 'mean': mean, 'std': std}
    elif type == 'xaiver': # xaiver(glorot) initializer
        mode = get_value('mode', default='IN')
        if mode == 'IN':
            mode = 'FAN_IN'
        elif mode == 'OUT':
            mode = 'FAN_OUT'
        elif mode == 'AVG':
            mode == 'AVERAGE'
        init = {'type': type, 'variance_norm': mode}
    elif type == 'msra': # delving deep into rectifiers(MSRA) initializer
        mode = get_value('mode', default='IN')
        if mode == 'IN':
            mode = 'FAN_IN'
        elif mode == 'OUT':
            mode = 'FAN_OUT'
        elif mode == 'AVG':
            mode == 'AVERAGE'
        init = {'type': type, 'variance_norm': mode}
    elif type == 'bilinear': # bilinear initializer
        init = {'type': type}
    elif type == 'positive_unitball': # positive unitball initializer
        init = {'type': type}
    else: # TODO: error control
        init = None
    return init



