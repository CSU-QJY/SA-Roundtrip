# import numpy as np
#
# from ops import *
# from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten
#
# from tensorflow_addons.layers import SpectralNormalization
#
# weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
#
#
# def BigBiGAN_G(input_shape=60, output_shape=32, out_ch=1, weight_init=weight_init, name='generator', dim=64):
#     inputs_1 = tf.keras.Input(shape=input_shape)
#
#     ch = dim * 4
#     n_layers = int(np.log2(output_shape) - 2)
#
#     dense0 = SpectralNormalization(Dense(4 * 4 * ch, use_bias=True, kernel_initializer=weight_init))
#
#     conv0 = SpectralNormalization(Conv2DTranspose(out_ch, 3, 1, padding='SAME', use_bias=False,
#                                                   kernel_initializer=weight_init))
#
#     z_split = tf.split(inputs_1, num_or_size_splits=[input_shape // (n_layers + 1)] * (n_layers + 1), axis=-1)
#
#     # Fully connected
#     l = dense0(z_split[0])  # 4*4*4*ch
#     l = tf.reshape(l, shape=[-1, 4, 4, ch])  # [-1, 4, 4, 4*ch]
#     l = resblock_up_condition_top(l, z_split[1], channels=ch,
#                                   weight_init=weight_init, use_bias=False)
#     for n in range(1, n_layers):
#         l = resblock_up_condition(l, z_split[n + 1], channels=ch // 2 ** n, weight_init=weight_init, use_bias=False)
#         if l.shape[1] == output_shape // 2:
#             l = google_attention(l, l.shape[-1])
#
#     l = tf.keras.layers.BatchNormalization()(l)
#     l = tf.keras.layers.ReLU()(l)
#
#     # Output layer
#     output = conv0(l)
#     output = tf.nn.sigmoid(output)
#     return tf.keras.Model(inputs=inputs_1, outputs=output, name=name)
#
#
# def BigBiGAN_D_F(input_shape=None, output_shape=32, weight_init=weight_init, name='discriminator_f', dim=64):
#     if input_shape is None:
#         input_shape = [32, 32, 1]
#
#     n_layers = int(np.log2(output_shape) - 2)
#     dense0 = Dense(units=1, use_bias=True, kernel_initializer=weight_init)
#
#     inputs = tf.keras.Input(shape=input_shape)
#     l = resblock_down(inputs, channels=dim, weight_init=weight_init, use_bias=False)
#
#     for n in range(1, n_layers):
#         if l.shape[1] == output_shape // 2:
#             l = google_attention(l, l.shape[-1])
#         l = resblock_down(l, channels=dim * 2 ** n, weight_init=weight_init, use_bias=False)
#
#     l = resblock(l, channels=dim * 2 ** (n_layers - 1), weight_init=weight_init, use_bias=False)
#     l = tf.keras.layers.ReLU()(l)
#     l = tf.math.reduce_sum(l, axis=[1, 2])
#     output2 = dense0(l)
#
#     return tf.keras.Model(inputs=inputs, outputs=output2, name=name)
#
#
# def BigBiGAN_D_H(input_shape=60, output_shape=32, weight_init=weight_init, training=False, name='discriminator_h',
#                  dim=64):
#     flatten = Flatten()
#     dense0 = Dense(units=1, use_bias=True, kernel_initializer=weight_init)
#     ch = dim // 1
#     inputs = tf.keras.Input(shape=input_shape)
#     l1 = flatten(inputs)
#     l1 = resblock_dense(l1, units=ch, weight_init=weight_init, training=training)
#     l2 = resblock_dense(l1, units=ch, weight_init=weight_init)
#
#     output1 = resblock_dense(l2, units=ch, weight_init=weight_init)
#
#     output2 = dense0(output1)
#
#     return tf.keras.Model(inputs=inputs, outputs=output2, name=name)
#
#
# def BigBiGAN_E(input_shape=None, output_shape=60, out_ch=1, weight_init=None, name='encoder', dim=32):
#     if input_shape is None:
#         input_shape = [32, 32, 1]
#     ch = dim // 2
#     n_layers = int(np.log2(input_shape[1]) - 2)
#     inputs = tf.keras.Input(shape=input_shape)
#
#     upsample = tf.keras.layers.UpSampling2D(2)
#     conv0 = Conv2D(ch, (7, 7), strides=2, padding='SAME', kernel_initializer=weight_init)
#     dense0 = Dense(units=output_shape, kernel_initializer=weight_init)
#     flatten = Flatten()
#     maxpooling0 = tf.keras.layers.AveragePooling2D((3, 3), strides=2)
#     l = upsample(inputs)
#     l = tf.reshape(l, (-1, input_shape[1] * 2, input_shape[1] * 2, out_ch))
#     l = conv0(l)
#     l = maxpooling0(l)
#     for n in range(n_layers - 1):
#         l = bottleneck_rev_s(l, ch=ch * 2 ** n, weight_init=weight_init)
#         l = bottleneck_rev_s(l, ch=ch * 2 ** n, weight_init=weight_init)
#         l = pool_and_double_channels(l, 2)
#     l = bottleneck_rev_s(l, ch=ch * 2 ** (n_layers - 1), weight_init=weight_init)
#     l = bottleneck_rev_s(l, ch=ch * 2 ** (n_layers - 1), weight_init=weight_init)
#     l = tf.keras.layers.GlobalAveragePooling2D()(l)
#     l = flatten(l)
#     l = resblock_dense_no_sn(l, units=ch * 2 ** n_layers, weight_init=weight_init)
#     l = resblock_dense_no_sn(l, units=ch * 2 ** n_layers, weight_init=weight_init)
#     l = dense0(l)
#     return tf.keras.Model(inputs=inputs, outputs=l, name=name)
#
#
# if __name__ == '__main__':
#     G = BigBiGAN_G()
#     G.summary()
#     F = BigBiGAN_D_F()
#     F.summary()
#     H = BigBiGAN_D_H()
#     H.summary()
#     E = BigBiGAN_E()
#     E.summary()
from new_ops import *
from tensorflow.python.keras.layers import Input, Activation, Flatten, Reshape, UpSampling2D
import matplotlib.pyplot as plt
weight_init = tf.initializers.TruncatedNormal(stddev=0.02)
# weight_init = tf.initializers.glorot_uniform()


def Discriminator(input_dim=120, name='dx_net', nb_layers=4, nb_units=512):
    """x->input_shape,nb_layers->n_downsamplings,nb_units->dim"""
    inputs = Input(shape=input_dim)
    fc = Dense(nb_units, kernel_initializer=weight_init)(inputs)
    fc = resblock_dense(fc, nb_units, weight_init)
    for _ in range(nb_layers):
        fc = resblock_dense(fc, nb_units, weight_init)
    fc = BatchNormalization()(fc)
    fc = Activation('tanh')(fc)
    fc = Dense(1, kernel_initializer=weight_init)(fc)
    return tf.keras.Model(inputs=inputs, outputs=fc, name=name)


'''
判别器 D 使用了 Spectral norm 之后，就不能使用 BatchNorm (或者其它 Norm) 了。
原因也很简单，因为 Batch norm 的“除方差”和“乘以缩放因子”这两个操作很明显会破坏判别器的 Lipschitz 连续性。
'''


def Discriminator_img(input_dim=(32, 32, 1), name='dy_net', nb_layers=4, nb_units=32):

    model_input = Input(shape=input_dim)
    h = SpectralNormalization(Conv2D(nb_units, 7, strides=1, padding='same', kernel_initializer=weight_init))(model_input)
    h = LeakyReLU()(h)
    h = AveragePooling2D(pool_size=2, strides=2)(h)
    # h = resblock_sn(inputs=h, channels=nb_units, weight_init=weight_init)
    for n in range(nb_layers):
        if h.shape[1] == 16:
            h = google_attention(h, nb_units)
        nb_units *= 2
        h = resblock_down(inputs=h, channels=nb_units, weight_init=weight_init)
        # if n == nb_layers - 1:
        #     h = resblock_down(inputs=h, channels=nb_units, weight_init=weight_init)
    h = LeakyReLU()(h)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    model_output = Dense(1, kernel_initializer=weight_init)(h)
    return tf.keras.Model(inputs=model_input, outputs=model_output, name=name)


def Generator_img(input_dim=(1, 1, 128), name='G', nb_layers=2, nb_units=1024):
    model_input = Input(shape=input_dim)
    h = SpectralNormalization(
        Conv2DTranspose(nb_units, 4, 1, use_bias=False, padding='valid', kernel_initializer=weight_init))(model_input)
    for n in range(nb_layers):
        nb_units //= 2
        h = resblock_up(inputs=h, channels=nb_units, weight_init=weight_init, use_bias=False)
        # if h.shape[1]     == 16:
        #     h = google_attention(h, nb_units)
    h = FilterResponseNormalization()(h)
    h = TLU()(h)
    model_output = Conv2DTranspose(1, kernel_size=7, strides=1, padding='same', activation='tanh',
                                   kernel_initializer=weight_init)(h)
    return tf.keras.Model(inputs=model_input, outputs=model_output, name=name)


def Encoder_img(input_dim=(32, 32, 1), output_dim=(1, 1, 128), name='H', nb_layers=4, nb_units=64):
    model_input = Input(shape=input_dim)
    h = Conv2D(nb_units, 7, strides=1, padding='same', use_bias=False, kernel_initializer=weight_init)(model_input)
    h = FilterResponseNormalization()(h)
    h = TLU()(h)
    h = AveragePooling2D(pool_size=2, strides=2)(h)
    # h = resblock(inputs=h, channels=nb_units, weight_init=weight_init)
    for n in range(nb_layers):
        nb_units *= 2
        h = resblock_en(h, nb_units, weight_init=weight_init, use_bias=False)
        # if n == nb_layers - 1:
        #     h = resblock_en(h, nb_units, weight_init=weight_init, use_bias=False)
    h = FilterResponseNormalization()(h)
    h = TLU()(h)
    model_output = Conv2D(128, 4, 1, padding='valid', kernel_initializer=weight_init)(h)

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
    h_net = Encoder_img(input_dim=(32, 32, 1), output_dim=(1, 1, 128), name='h_net', nb_layers=2, nb_units=64)
    h_net.summary()
    dx_net = Discriminator(input_dim=x_dim, name='dx_net', nb_layers=2, nb_units=128)
    dx_net.summary()
    dy_net = Discriminator_img(input_dim=(32, 32, 1), name='dy_net', nb_layers=2, nb_units=64)
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
