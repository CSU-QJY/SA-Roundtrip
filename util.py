from tensorflow.python.keras.datasets.cifar import load_batch
import tensorflow as tf
import h5py
import tf2lib as tl
import os
from skimage.exposure import rescale_intensity
import numpy as np
from tensorflow.python.keras import backend as K


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = tf.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_losses_fn():
    def disc_loss(real_f_output, real_h_output, fake_f_output, fake_h_output):
        real_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(real_f_output) - real_f_output)
                                   + tf.nn.relu(tf.ones_like(real_h_output) - real_h_output))

        fake_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(fake_f_output) + fake_f_output)
                                   + tf.nn.relu(tf.ones_like(fake_h_output) + fake_h_output))
        total_loss = real_loss + fake_loss
        return total_loss

    def gen_en_loss(real_f_output, real_h_output, fake_f_output, fake_h_output):
        real_loss = tf.reduce_mean(tf.reduce_sum([real_f_output, real_h_output], axis=0))
        fake_loss = tf.reduce_mean((-1) * tf.reduce_sum([fake_f_output, fake_h_output], axis=0))

        return real_loss + fake_loss

    return disc_loss, gen_en_loss


def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)

    return gp


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
                    idx = tf.random.uniform(shape=(), minval=0, maxval=len(self.items), dtype=tf.int32)
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)


def load_mnist(dataset, batch_size, drop_remainder=False, shuffle=True, repeat=1):
    if dataset == 'mnist':
        with np.load('mnist.npz', allow_pickle=True) as f:
            train_images, train_label = f['x_train'], f['y_train']

        train_images.shape = train_images.shape + (1,)
    elif dataset == 'fashion_mnist':
        (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, _), (_, _) = load_cifar()
    else:
        raise NotImplementedError

    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [32, 32])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5-1
        return img

    # train_images = _map_fn(train_images)
    dataset = tl.memory_data_batch_dataset(train_images,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=shuffle,
                                           repeat=repeat)
    img_shape = (32, 32, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset


def load_cifar():
    dirname = 'cifar-10-batches-py'

    path = os.path.join(dirname)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


def load_data(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1):
    f = h5py.File(dataset, 'r')

    train_images = f['ct_slices']
    # slice_class = f['slice_class']

    train_images = np.array(train_images)
    train_images = rescale_intensity(train_images, out_range=(0, 255))
    # slice_class = np.array(slice_class)
    # train_images = np.clip(train_images, -1000, 320)
    train_images = np.expand_dims(train_images, -1)

    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [64, 64])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.memory_data_batch_dataset(train_images,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=shuffle,
                                           repeat=repeat)
    img_shape = (64, 64, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset


def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1):
    if dataset == 'mnist':
        with np.load('mnist.npz', allow_pickle=True) as f:
            train_images, train_label = f['x_train'], f['y_train']

        train_images.shape = train_images.shape + (1,)
    elif dataset == 'fashion_mnist':
        (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, _), (_, _) = load_cifar()
    else:
        raise NotImplementedError

    @tf.function
    def _map_fn(img):
        img['data'] = tf.image.resize(img['data'], [32, 32])
        img['data'] = tf.clip_by_value(img['data'], 0, 255)
        img['data'] = img['data'] / 255
        return img['data']

    dataset = tl.memory_data_batch_dataset(train_images,
                                           train_label,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=shuffle,
                                           repeat=repeat)
    img_shape = (32, 32, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset


def make_28x28_dataset(batch_size):
    (train_images, train_label), (_, _) = load_mnist()
    indx = np.argwhere(train_label == 4)
    images = (np.reshape(train_images[indx], [-1, 784]) / 127.5 - 1).astype('float32')

    # indx=np.random.randint(low = 0, high = train_label.shape[0], size = batch_size)
    # images=np.reshape(train_images[indx],[batch_size,784])/127.5-1
    # label= np.reshape(np.eye(10)[train_label][indx],[batch_size,10])

    return images, train_label, len(images)


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255  # or img = tl.minmax_norm(img)

            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size,
                                        crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255  # or img = tl.minmax_norm(img)

            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = None  # cycle both
    else:
        A_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=A_repeat)

    A_dataset = tf.data.Dataset.zip((A_dataset))
    len_dataset = len(A_img_paths) // batch_size

    return A_dataset, len_dataset

if __name__ == '__main__':
    # batch_size=64
    # ii=0
    # dataset,len_dataset=make_28x28_dataset(batch_size)
    # for inx in range(930):
    #
    #     i = ii + batch_size
    #     if i < len_dataset:
    #         x_real = dataset[ii:i, :]
    #         ii = ii + batch_size
    #         print(ii)
    #     else:
    #         x_real = np.concatenate((dataset[ii:len_dataset, :], dataset[(i - len_dataset):, :]), axis=0)
    #         ii=i - len_dataset
    #         print(ii)
    load_cifar()
