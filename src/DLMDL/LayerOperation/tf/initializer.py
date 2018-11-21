import tensorflow as tf

# TODO: Class tf.initializers are not defined in version 1.1.0. We replace it with another API
def get_initializer(initializer, is_bias=False  ):
    """
    get tensorflow initializer method.
    :param initializer: DLMDL weight/bias initializer declaration. Dictionary
    :param is_bias: whether bias or not
    :return: weight/bias initializer method
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
        value = get_value('value', default=0.0)
        if value is None:
            raise Exception('[DLMDL ERROR]: {0} in uniform initializer must be declared.'.format('value'))
        init = tf.constant_initializer(value=value, dtype=tf.float32)
    elif type == 'uniform': # uniform random initializer
        max = get_value('max', default=None)
        if max is None:
            raise Exception('[DLMDL ERROR]: {0} in uniform initializer must be declared.'.format('max'))
        min = get_value('min', default=0.0)
        seed = get_value('seed', default=None)
        init = tf.random_uniform_initializer(minval=min, maxval=max, seed=seed, dtype=tf.float32)
    elif type == 'normal': # normal initializer
        mean = get_value('mean', default=0.0)
        std = get_value('std', default=0.01)
        seed = get_value('seed', default=None)
        init = tf.random_normal_initializer(mean=mean, stddev=std, seed=seed, dtype=tf.float32)
    elif type == 'xaiver': # xaiver(glorot) initializer
        mode = get_value('mode', default='IN')
        factor = get_value('factor', default=3.0)
        dist = get_value('dist', default='UNIFORM')
        seed = get_value('seed', default=None)
        is_uniform = True if dist == 'UNIFORM' else False
        init = tf.contrib.layers.variance_scaling_initializer(factor=factor,
                                                              mode='FAN_' + mode,
                                                              uniform=is_uniform,
                                                              seed=seed, dtype=tf.float32)
    elif type == 'msra': # delving deep into rectifiers(MSRA) initializer
        mode = get_value('mode', default='IN')
        seed = get_value('seed', default=None)
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                              mode='FAN_' + mode,
                                                              uniform=True,
                                                              seed=seed, dtype=tf.float32)
    elif type == 'orthogonal': # orthogonal matrix initializer
        factor = get_value('factor', default=1.0)
        seed = get_value('seed', default=None)
        init = tf.orthogonal_initializer(gain=factor, seed=seed, dtype=tf.float32)
    elif type == 'identity': # TODO: not supported in TF v1.1.0. TO be updated
        pass
    elif type == 'truncated_normal': # truncated normal initializer
        mean = get_value('mean', default=0.0)
        std = get_value('std', default=0.01)
        seed = get_value('seed', default=None)
        init = tf.truncated_normal_initializer(mean=mean, std=std, seed=seed, dtype=tf.float32)
    else: # TODO: error control
        init = None
    return init



