import tensorflow as tf

class LeNet(object):
    def __init__(self, activation='relu'):
        self.activation = activation
        self.build_model()

    TOWER_NAME = 'tower'
    def _activation_summary(x):
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                                             tf.nn.zero_fraction(x))


    def _variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float16
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var


    def _variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float16
        var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return var

    def build_model(self):
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape = [5, 5, 1, 6],
                                                 stddev = 5e-2,
                                                 wd = 0.0)
            conv = tf.nn.conv2d
