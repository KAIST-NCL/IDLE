
import tensorflow as tf

def get_regularizer(regularizer, scope, is_bias=False):
    """
    get tensorflow regularizer method
    :param regularizer: DLMDL weight/bias regularizer declaration. Dictionary
    :param scope: weight scope
    :param is_bias: whether bias or not
    :return: weight/bias regularizer method
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
    if type == 'L2':
        reg = tf.contrib.layers.l2_regularizer(scale=decay, scope=scope)
    elif type == 'L1':
        reg = tf.contrib.layers.l1_regularizer(scale=decay, scope=scope)
    elif type == 'L1_L2':
        reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=decay, scale_12=decay, scope=scope)
    else: # TODO: error control
        reg = None
    return reg