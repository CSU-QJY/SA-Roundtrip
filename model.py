import tensorflow as tf
from tensorflow import keras
import numpy as np





def Discriminator(input_dim=120, name='dx_net', nb_layers=2, nb_units=1024):
    """x->input_shape,nb_layers->n_downsamplings,nb_units->dim"""
    x = inputs = keras.Input(shape=input_dim)
    fc = keras.layers.Dense(nb_units)(x)
    fc = tf.nn.leaky_relu(fc)
    for _ in range(nb_layers-1):
        # nb_units //= 2
        fc = keras.layers.Dense(nb_units)(fc)
        fc = keras.layers.BatchNormalization()(fc)
        fc = tf.nn.tanh(fc)

    fc = keras.layers.Dense(1)(fc)
    return keras.Model(inputs=inputs, outputs=fc, name=name)


def Discriminator_img(input_dim=32 * 32, name='dy_net', nb_layers=4, nb_units=64):
    z = inputs = keras.Input(shape=input_dim)
    dim = np.sqrt(input_dim).astype('int32')
    x = tf.reshape(z, [-1, dim, dim, 1])

    conv = keras.layers.Conv2D(nb_units, 4, strides=2, padding='SAME')(x)
    conv = tf.nn.leaky_relu(conv)

    for _ in range(nb_layers-1):
        nb_units *= 2
        conv = keras.layers.Conv2D(nb_units, 4, strides=(2, 2), padding='SAME')(conv)
        conv = keras.layers.BatchNormalization(momentum=0.9, scale=True)(conv)
        conv = tf.nn.leaky_relu(conv)
    fc = keras.layers.Flatten()(conv)

    fc = tf.keras.layers.Dense(1)(fc)
    return keras.Model(inputs=inputs, outputs=fc, name=name)


def Generator_img(input_dim=120, name='G', nb_layers=2, nb_units=512):
    z = inputs = keras.Input(shape=input_dim)

    fc = keras.layers.Dense(1024)(z)
    fc = keras.layers.BatchNormalization(momentum=0.9, scale=True)(fc)
    fc = tf.nn.leaky_relu(fc)
    fc = keras.layers.Dense(8 * 8 * 128)(fc)
    fc = tf.reshape(fc, [-1, 8, 8, 128])
    fc = keras.layers.BatchNormalization(momentum=0.9, scale=True)(fc)
    fc = tf.nn.leaky_relu(fc)

    conv = keras.layers.Convolution2DTranspose(nb_units, 4, strides=2, padding='SAME', use_bias=False)(fc)
    conv = keras.layers.BatchNormalization(momentum=0.9, scale=True)(conv)
    conv = tf.nn.leaky_relu(conv)
    for _ in range(nb_layers-1):
        nb_units //= 2
        conv = keras.layers.Convolution2DTranspose(nb_units, 4, strides=2, padding='SAME', use_bias=False)(conv)
        conv = keras.layers.BatchNormalization(momentum=0.9, scale=True)(conv)
        conv = tf.nn.leaky_relu(conv)
    conv = keras.layers.Conv2DTranspose(1, 4, 1, padding='SAME')(conv)
    conv = tf.nn.sigmoid(conv)
    conv = tf.reshape(conv, [-1, 32 * 32])
    return keras.Model(inputs=inputs, outputs=conv, name=name)


def Encoder_img(input_dim=32 * 32, output_dim=50, name='H', nb_layers=4, nb_units=64):
    z = inputs = keras.Input(shape=input_dim)
    dim = np.sqrt(input_dim).astype('int32')
    z = tf.reshape(z, [-1, dim, dim, 1])
    z = keras.layers.UpSampling2D(2)(z)
    conv = keras.layers.Conv2D(nb_units, 4, 2, padding='SAME')(z)
    conv = tf.nn.leaky_relu(conv)
    for _ in range(nb_layers):
        nb_units *= 2
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

    x_dim = 50
    y_dim = 32 * 32
    # batch_size = 64
    # dataset, train_label, len_dataset = util.make_28x28_dataset(batch_size)
    # x = tf.random.normal(shape=(batch_size, x_dim))
    # indx = np.random.randint(low=0, high=dataset.shape[0], size=batch_size)
    # data = (np.reshape(dataset[indx], [batch_size, 784]) / 255)
    # label = np.reshape(np.eye(10)[train_label][indx], [batch_size, 10])
    g_net = Generator_img(input_dim=x_dim, name='g_net', nb_layers=2, nb_units=64)
    h_net = Encoder_img(input_dim=y_dim, output_dim=x_dim, name='h_net', nb_layers=2, nb_units=32)
    dx_net = Discriminator(input_dim=x_dim, name='dx_net', nb_layers=2, nb_units=64)
    dy_net = Discriminator_img(input_dim=y_dim, name='dy_net', nb_layers=2, nb_units=64)
    print()
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
