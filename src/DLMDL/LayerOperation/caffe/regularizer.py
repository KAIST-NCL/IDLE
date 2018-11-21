def get_regularizer(regularizer, is_bias=False):
    """
    get caffe regularizer method
    :param regularizer: DLMDL weight/bias regularizer declaration. Dictionary
    :param is_bias: whether bias or not
    :return: weight/bias regularizer dictionary
    """
    def get_value(key, default=None):
        """
        get attributes in regularizer
        :param key: key value for parsing
        :param default: default value of attribute
        :return: value
        """
        value = regularizer.get(key)
        if value is None:
            return default
        return value

    # get regularizer type
    # default: L2 regularizer
    type = get_value('type', default='L2')
    decay = float(get_value('bias', default=0.0)) if is_bias else float(get_value('weight', default=0.0))
    reg = {'decay_mult': decay}
    return reg, type