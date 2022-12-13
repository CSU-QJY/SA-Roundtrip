import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization,LeakyReLU
from tensorflow_addons.layers import SpectralNormalization,FilterResponseNormalization,TLU
from tensorflow_addons.utils import types

weight_regularizer = None  # orthogonal_regularizer(0.0001)
weight_regularizer_fully = None


def try_Attention(inputs, ch):
    b, h, w, c = inputs.shape
    filters_f_g_h = ch // 8
    filters_v = ch
    gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)
    query = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
    key = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
    value = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=filters_v, kernel_size=1, strides=1, use_bias=True))
    o = tf.keras.layers.Attention()([hw_flatten(key(inputs)), hw_flatten(value(inputs)), hw_flatten(query(inputs))])

    o = tf.reshape(o, shape=[-1, h, w, c])
    o = value(o, training=True)

    output = gamma * o + inputs
    return output

def google_attention(inputs, ch):
    b, h, w, c = inputs.shape
    gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    query = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True))(inputs)
    query = tf.keras.layers.MaxPooling2D()(query)

    key = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True))(inputs)

    value = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch // 2, kernel_size=1, strides=1, use_bias=True))(inputs)
    value = tf.keras.layers.MaxPooling2D()(value)

    o = tf.keras.layers.Attention()([hw_flatten(key), hw_flatten(value), hw_flatten(query)])

    o = tf.reshape(o, shape=[-1, h, w, c//2])
    o = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch, kernel_size=1, strides=1, use_bias=True))(o)
    output = gamma * o + inputs
    return output

def google_attention_no_sn(inputs, ch):
    b, h, w, c = inputs.shape
    gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    query = tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True)(inputs)
    query = tf.keras.layers.MaxPooling2D()(query)

    key = tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True)(inputs)

    value = tf.keras.layers.Conv2D(filters=ch // 2, kernel_size=1, strides=1, use_bias=True)(inputs)
    value = tf.keras.layers.MaxPooling2D()(value)

    o = tf.keras.layers.Attention()([hw_flatten(key), hw_flatten(value), hw_flatten(query)])

    o = tf.reshape(o, shape=[-1, h, w, c//2])
    o = tf.keras.layers.Conv2D(filters=ch, kernel_size=1, strides=1, use_bias=True)(o)
    output = gamma * o + inputs
    return output

def hw_flatten(x):
    b, h, w, c = x.shape
    return tf.reshape(x, shape=[-1, h * w, c])


def resblock(inputs, channels, weight_init, use_bias=True):
    conv0 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init))
    conv1 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init))

    x = tf.keras.layers.ReLU()(inputs)
    x = conv0(x)
    # res2
    x = tf.keras.layers.ReLU()(x)
    x = conv1(x)

    return tf.keras.layers.Add()([x, inputs])


def resblock_up_condition_top(inputs, z, channels, weight_init, use_bias=True):
    conv1 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer))

    conv2 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer))
    skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                             kernel_size=1,
                                             strides=1,
                                             padding='SAME',
                                             use_bias=use_bias,
                                             kernel_initializer=weight_init,
                                             kernel_regularizer=weight_regularizer))
    upsampling = tf.keras.layers.UpSampling2D()
    cbn = condition_batch_norm
    x = cbn(channels=channels, weight_init=weight_init)([inputs, z])
    x = tf.nn.relu(x)
    x = conv1(x)
    x = upsampling(x)


    # res2
    x = cbn(channels=channels, weight_init=weight_init)([x, z])
    x = tf.nn.relu(x)
    x = conv2(x)

    # skip
    x_init = upsampling(inputs)
    x_init = skip_conv(x_init)

    return tf.keras.layers.Add()([x, x_init])


def resblock_up_condition(inputs, z, channels, weight_init, use_bias=True):
    conv1 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer))

    conv2 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer))
    skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                             kernel_size=1,
                                             strides=1,
                                             padding='SAME',
                                             use_bias=use_bias,
                                             kernel_initializer=weight_init,
                                             kernel_regularizer=weight_regularizer))
    upsampling = tf.keras.layers.UpSampling2D()
    cbn = condition_batch_norm
    x = cbn(channels=channels * 2, weight_init=weight_init)([inputs, z])
    x = tf.nn.relu(x)
    x = upsampling(x)
    x = conv1(x)

    # res2
    x = cbn(channels=channels, weight_init=weight_init)([x, z])
    x = tf.nn.relu(x)
    x = conv2(x)

    # skip
    x_init = upsampling(inputs)
    x_init = skip_conv(x_init)

    return tf.keras.layers.Add()([x, x_init])


def resblock_down(inputs, channels, weight_init, use_bias=True):
    conv0 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init))
    conv1 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=use_bias,
                                         kernel_initializer=weight_init))
    skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                             kernel_size=1,
                                             strides=1,
                                             use_bias=use_bias,
                                             kernel_initializer=weight_init))
    avg_pooling = tf.keras.layers.AveragePooling2D()
    x = tf.keras.layers.ReLU()(inputs)
    x = conv0(x)
    # res2
    x = tf.keras.layers.ReLU()(x)
    x = conv1(x)
    x = avg_pooling(x)
    # skip
    x_init = skip_conv(inputs)
    x_init = avg_pooling(x_init)

    return tf.keras.layers.Add()([x, x_init])


def resblock_dense(inputs, units, weight_init, training=True):
    dense0 = SpectralNormalization(Dense(units=units, kernel_initializer=weight_init))
    dropout = tf.keras.layers.Dropout(0.2)
    dense1 = SpectralNormalization(Dense(units=units, kernel_initializer=weight_init))
    dense_skip = SpectralNormalization(Dense(units=units, kernel_initializer=weight_init))

    l1 = dense0(inputs)
    l1 = dropout(l1, training=training)
    l1 = LeakyReLU()(l1)

    l2 = dense1(l1)
    l2 = dropout(l2, training=training)
    l2 = LeakyReLU()(l2)

    skip = dense_skip(inputs)
    skip = tf.nn.leaky_relu(skip)

    output = l2 + skip

    return output


def resblock_dense_no_sn(inputs, units, weight_init):
    dense0 = Dense(units=units, kernel_initializer=weight_init)
    dropout = tf.keras.layers.Dropout(0.2)
    dense1 = Dense(units=units, kernel_initializer=weight_init)
    dense_skip = Dense(units=units, kernel_initializer=weight_init)

    l1 = dense0(inputs)
    l1 = dropout(l1)
    l1 = tf.keras.layers.LeakyReLU()(l1)

    l2 = dense1(l1)
    l2 = dropout(l2)
    l2 = tf.keras.layers.LeakyReLU()(l2)

    skip = dense_skip(inputs)
    skip = tf.keras.layers.LeakyReLU()(skip)

    output = l2 + skip

    return output


def bottleneck_s(inputs, filters, weight_init, strides=1):
    conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False,
                                   kernel_initializer=weight_init)

    conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', strides=strides,
                                   use_bias=False, kernel_initializer=weight_init)
    skip_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='SAME',
                                       kernel_initializer=weight_init)
    l1 = conv0(inputs)
    l1 = tf.keras.layers.experimental.SyncBatchNormalization()(l1)
    l1 = tf.keras.layers.LeakyReLU()(l1)

    l2 = conv1(l1)
    l2 = tf.keras.layers.experimental.SyncBatchNormalization()(l2)
    l2 = tf.keras.layers.LeakyReLU()(l2)

    # Project input if necessary
    if (strides > 1) or (filters != inputs.shape[-1]):
        x_shortcut = skip_conv(inputs)
    else:
        x_shortcut = inputs

    return l2 + x_shortcut


def bottleneck_rev_s(inputs, ch, weight_init, strides=1):
    x1, x2 = tf.split(inputs, 2, 3)
    y1 = x1 + bottleneck_s(x2, filters=ch // 2, strides=strides, weight_init=weight_init)

    return tf.concat([x2, y1], axis=3)


def pool_and_double_channels(x, pool_stride):
    if pool_stride > 1:
        x = tf.nn.avg_pool2d(x, ksize=pool_stride,
                             strides=pool_stride,
                             padding='SAME')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0],
                      [x.get_shape().as_list()[3] // 2, x.get_shape().as_list()[3] // 2]])


@tf.keras.utils.register_keras_serializable(package='Addons')
class condition_batch_norm(tf.keras.layers.Layer):

    def __init__(
            self,
            channels: int = 1024,
            epsilon: float = 1e-5,
            weight_init=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.channels = channels
        self.weight_init = weight_init
        self.epsilon = epsilon

    def build(self, input_shape):
        self._add_dense()
        self._add_mean_weight(input_shape)
        self._add_var_weight(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        beta0 = self.beta0(inputs[-1])
        gamma0 = self.gamma0(inputs[-1])
        beta = tf.reshape(beta0, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma0, shape=[-1, 1, 1, self.channels])
        if training:
            batch_mean, batch_var = tf.nn.moments(inputs[0], [0, 1, 2])
            self.test_mean.assign(self.test_mean * 0.9999 + batch_mean * 0.0001)
            self.test_var.assign(self.test_var * 0.9999 + batch_var * 0.0001)

            return tf.nn.batch_normalization(inputs[0], batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(inputs[0], self.test_mean, self.test_var, beta, gamma, self.epsilon)

    def get_config(self):
        config = {
            "channels": self.channels,
            "weight_init": self.weight_init,
            "epsilon": self.epsilon
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _add_mean_weight(self, input_shape):
        self.test_mean = tf.Variable(tf.zeros(input_shape[0][-1]), dtype=tf.float32, trainable=False, name="mean")

    def _add_var_weight(self, input_shape):
        self.test_var = tf.Variable(tf.ones(input_shape[0][-1]), dtype=tf.float32, trainable=False, name="var")

    def _add_dense(self):
        self.beta0 = tf.keras.layers.Dense(units=self.channels, use_bias=True, kernel_initializer=self.weight_init)
        self.gamma0 = tf.keras.layers.Dense(units=self.channels, use_bias=True, kernel_initializer=self.weight_init)

@tf.keras.utils.register_keras_serializable(package='Addons')
class PositionalNormalization(tf.keras.layers.Layer):
    """Positional normalization removes the mean and standard deviation over spatial dimensions."""

    def __init__(self, epsilon=1e-5, axis=3):
        """Sets parameters of the layer.
        Args:
            epsilon: a small value to stabilize the computations
            axis: the axis along which to compute the moments (the default applies to channels in BxHxWxC)
        """
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis

    def call(self, inputs):
        """
        Args:
            inputs:
        Returns:
            The input with the moments removed, and the computed mean and standard deviation that can be passed
            to a later `MomentShortcut` layer.
        """
        mean, variance = tf.nn.moments(inputs, axes=[self.axis], keepdims=True)
        standard_deviation = tf.sqrt(variance + self.epsilon)
        output = (inputs - mean) / standard_deviation
        return output, mean, standard_deviation