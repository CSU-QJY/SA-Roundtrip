import tensorflow as tf
from tensorflow import keras
import copy
import _pickle as pickle
import gzip
import numpy as np


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    return tf.concat([x, y * tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(y)[3]])], 3)


def Discriminator(x=100, input_dim=100, name='dx_net', nb_layers=2, nb_units=256):
    """x->input_shape,nb_layers->n_downsamplings,nb_units->dim"""
    x = inputs = keras.Input(shape=x)
    fc = keras.layers.Dense(nb_units)(x)
    fc = tf.nn.leaky_relu(fc)
    for _ in range(nb_layers - 1):
        fc = keras.layers.Dense(nb_units)(fc)
        fc = keras.layers.BatchNormalization()(fc)

        # fc = leaky_relu(fc)
        fc = tf.nn.tanh(fc)
    fc = keras.layers.Dense(1)(fc)
    return keras.Model(inputs=inputs, outputs=fc, name=name)


def Discriminator_img(z=[784], input_dim=784, name='dy_net', nb_layers=2, nb_units=256):
    z = inputs = keras.Input(shape=tf.TensorShape([None]))
    x = z[:, :input_dim]
    # y = z[:, input_dim:]
    # y = tf.reshape(y, shape=[-1, 10])
    # y.set_shape([None, 10])
    # yb = tf.reshape(y, shape=[-1, 1, 1, 10])
    x = tf.reshape(x, [-1, 28, 28, 1])
    # x = conv_cond_concat(x, yb)
    conv = keras.layers.Conv2D(32, 4, strides=2, padding='SAME')(x)
    conv = tf.nn.leaky_relu(conv)
    # conv = conv_cond_concat(conv, yb)
    for _ in range(nb_layers - 1):
        conv = keras.layers.Conv2D(64, 4, strides=(2, 2),padding='SAME')(conv)
        conv = keras.layers.BatchNormalization(momentum=0.9, scale=True)(conv)
        conv = tf.nn.leaky_relu(conv)
    fc = keras.layers.Flatten()(conv)
    # fc = tf.concat([fc, y], axis=1)
    fc = keras.layers.Dense(1024)(fc)
    fc = keras.layers.BatchNormalization(momentum=0.9, scale=True)(fc)
    fc = tf.nn.leaky_relu(fc)
    # fc = tf.concat([fc, y], axis=1)
    fc = tf.keras.layers.Dense(1)(fc)
    return keras.Model(inputs=inputs, outputs=fc, name=name)


def Generator_img(z=[100], input_dim=100, output_dim=784, name='G', nb_layers=2, nb_units=256):
    z = inputs = keras.Input(shape=z)
    # inp=inputs=tf.reshape(z,[-1,tttt])
    # bs = tf.shape(z)[0]
    # y = z[:, -10:]
    # yb = tf.reshape(y, shape=[-1, 1, 1, 10])
    # ybs = yb[:, :, :, :]
    fc = keras.layers.Dense(1024)(z)
    fc = keras.layers.BatchNormalization(momentum=0.9, scale=True)(fc)
    fc = tf.nn.leaky_relu(fc)
    # fc = tf.concat([fc, y], 1)
    fc = keras.layers.Dense(7 * 7 * 128)(fc)
    fc = tf.reshape(fc, [-1, 7, 7, 128])
    # fc = tf.reshape(fc, tf.stack([bs, 7, 7, 128]))
    fc = keras.layers.BatchNormalization(momentum=0.9, scale=True)(fc)
    fc = tf.nn.leaky_relu(fc)
    # fc=tf.concat([fc, yb], 3)
    # fc = conv_cond_concat(fc, yb)
    # fc = tf.reshape(fc,[-1,7,7,138])
    conv = keras.layers.Convolution2DTranspose(64, 4, strides=2, padding='SAME', use_bias=False)(fc)
    conv = keras.layers.BatchNormalization(momentum=0.9, scale=True)(conv)
    conv = tf.nn.leaky_relu(conv)
    conv = keras.layers.Conv2DTranspose(1, 4, 2, padding='SAME')(conv)
    conv = tf.nn.tanh(conv)
    conv = tf.reshape(conv, [-1, 784])
    return keras.Model(inputs=inputs, outputs=conv, name=name)


def Encoder_img(x=784, input_dim=784, output_dim=100, name='H', nb_layers=2, nb_units=256, cond=True):
    x = inputs = keras.Input(shape=x)
    x = tf.reshape(x, [-1, 28, 28, 1])
    conv = keras.layers.Conv2D(64, 4, 2, padding='SAME')(x)
    conv = tf.nn.leaky_relu(conv)
    for _ in range(nb_layers - 1):
        conv = keras.layers.Conv2D(nb_units, 4, 2, padding='SAME')(conv)
        conv = tf.nn.leaky_relu(conv)
    conv = keras.layers.Flatten()(conv)
    fc = keras.layers.Dense(1024)(conv)
    fc = keras.layers.Dense(output_dim)(fc)

    return keras.Model(inputs=inputs, outputs=fc, name=name)


if __name__ == '__main__':
    import numpy as np
    import time
    import util
    from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian

    x_dim = 100
    y_dim = 784
    batch_size = 64
    dataset, train_label, len_dataset = util.make_28x28_dataset(batch_size)
    x = tf.random.normal(shape=(batch_size, x_dim))
    indx = np.random.randint(low=0, high=dataset.shape[0], size=batch_size)
    data = (np.reshape(dataset[indx], [batch_size, 784]) / 255)
    label = np.reshape(np.eye(10)[train_label][indx], [batch_size, 10])
    g_net = Generator_img(input_dim=x_dim, output_dim=y_dim, name='g_net', nb_layers=2, nb_units=256)
    h_net = Encoder_img(input_dim=y_dim, output_dim=x_dim, name='h_net', nb_layers=2, nb_units=256, cond=True)
    dx_net = Discriminator(input_dim=x_dim, name='dx_net', nb_layers=2, nb_units=128)
    dy_net = Discriminator_img(input_dim=y_dim, name='dy_net', nb_layers=2, nb_units=128)
    # y_ = g_net(x, training=True)  # G()
    #
    # x_ = h_net(data, training=True)  # H()
    # x__ = h_net(y_, training=True)  # H(G())
    # # x_combine_ = tf.concat([x_ , nb_classes],axis=1)
    # # y__ = g_net(x_combine_, training=True)  # G(H())
    # y__ = g_net(x_, training=True)  # G(H())
    # # dy_ = dy_net(tf.concat([y_ , nb_classes],axis=1), training=True)  # D(G())
    # dy_net.summary()
    # dy_ = dy_net(y_, training=True)  # D(G())
    # dx_ = dx_net(x_, training=True)  # D(H())
