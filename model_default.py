import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Conv2D, LeakyReLU, Conv2DTranspose
from tensorflow.python.keras.layers import Input, Activation, Flatten, BatchNormalization, Reshape

from new_ops import try_count_flops


def Discriminator(input_dim=120, name='dx_net', nb_layers=4, nb_units=512):
    """x->input_shape,nb_layers->n_downsamplings,nb_units->dim"""
    x = inputs = Input(shape=input_dim)
    fc = Flatten()(x)
    fc = Dense(nb_units)(fc)
    fc = LeakyReLU()(fc)

    fc = Dense(nb_units)(fc)
    fc = BatchNormalization()(fc)
    fc = Activation('tanh')(fc)
    fc = Dense(1)(fc)
    return tf.keras.Model(inputs=inputs, outputs=fc, name=name)


def Discriminator_img(input_dim=(32, 32, 1), name='dy_net', nb_layers=4, nb_units=32):
    model_input = Input(shape=input_dim)
    h = Reshape((32, 32, 1))(model_input)
    h = Conv2D(nb_units, 4, strides=2, padding='same')(h)
    h = LeakyReLU()(h)

    h = Conv2D(nb_units * 2, 4, 2, padding='same')(h)
    h = BatchNormalization(momentum=0.9, scale=True)(h)
    h = LeakyReLU()(h)
    h = Flatten()(h)
    h = Dense(1024)(h)
    h = BatchNormalization(momentum=0.9, scale=True)(h)
    h = LeakyReLU()(h)
    model_output = Dense(1)(h)
    return tf.keras.Model(inputs=model_input, outputs=model_output, name=name)


def Generator_img(input_dim=(1, 1, 128), name='G', nb_layers=2, nb_units=1024):
    model_input = Input(shape=input_dim)
    h = Flatten()(model_input)
    h = Dense(1024)(h)
    h = BatchNormalization(momentum=0.9, scale=True)(h)
    h = LeakyReLU()(h)
    h = Dense(8 * 8 * 128)(h)
    h = Reshape((8, 8, 128))(h)
    h = BatchNormalization(momentum=0.9, scale=True)(h)
    h = LeakyReLU()(h)
    h = Conv2DTranspose(64, 4, 2, padding='same')(h)
    h = BatchNormalization(momentum=0.9, scale=True)(h)
    h = LeakyReLU()(h)

    model_output = Conv2DTranspose(1, kernel_size=4, padding='same', strides=2, activation='tanh')(h)
    return tf.keras.Model(inputs=model_input, outputs=model_output, name=name)


def Encoder_img(input_dim=(32, 32, 1), output_dim=(1, 1, 128), name='H', nb_layers=4, nb_units=64):
    model_input = Input(shape=input_dim)
    h = Conv2D(64, 4, 2, padding='same')(model_input)
    h = LeakyReLU()(h)
    h = Conv2D(256, 4, 2, padding='same')(h)
    h = LeakyReLU()(h)
    h = Flatten()(h)
    h = Dense(1024)(h)
    model_output = Dense(128)(h)
    model_output = Reshape((1, 1, 128))(model_output)
    return tf.keras.Model(inputs=model_input, outputs=model_output, name=name)


if __name__ == '__main__':
    import numpy as np
    import time
    import util
    from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian

    x_dim = 60
    y_dim = 32 * 32
    batch_size = 64
    # dataset, train_label, len_dataset = util.make_28x28_dataset(batch_size)
    # x = tf.random.normal(shape=(batch_size, x_dim))
    # indx = np.random.randint(low=0, high=dataset.shape[0], size=batch_size)
    # data = (np.reshape(dataset[indx], [batch_size, 784]) / 255)
    # label = np.reshape(np.eye(10)[train_label][indx], [batch_size, 10])
    g_net = Generator_img(input_dim=(1, 1, 128), name='g_net', nb_layers=3, nb_units=256)
    g_net.summary()
    try_count_flops(g_net)
    h_net = Encoder_img(input_dim=(32, 32, 1), output_dim=(1, 1, 128), name='h_net', nb_layers=2, nb_units=32)
    h_net.summary()
    dx_net = Discriminator(input_dim=x_dim, name='dx_net', nb_layers=3, nb_units=128)
    dx_net.summary()
    dy_net = Discriminator_img(input_dim=(32, 32, 1), name='dy_net', nb_layers=2, nb_units=32)
    dy_net.summary()
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
