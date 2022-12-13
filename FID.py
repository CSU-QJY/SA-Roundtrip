import warnings

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import tf2lib as tl
import tqdm



def update_mean_cov(mean, cov, N, batch):
    batch_N = batch.shape[0]

    x = batch
    N += batch_N
    x_norm_old = batch - mean
    mean = mean + x_norm_old.sum(axis=0) / N
    x_norm_new = batch - mean
    cov = ((N - batch_N) / N) * cov + x_norm_old.T.dot(x_norm_new) / N

    return (mean, cov, N)


def frechet_distance(mean1, cov1, mean2, cov2):
    """Frechet distance between two multivariate Gaussians.
    Arguments:
        mean1, cov1, mean2, cov2: The means and covariances of the two
            multivariate Gaussians.
    Returns:
        The Frechet distance between the two distributions.
    """

    def check_nonpositive_eigvals(l):
        nonpos = (l < 0)
        if nonpos.any():
            warnings.warn('Rank deficient covariance matrix, '
                          'Frechet distance will not be accurate.', Warning)
        l[nonpos] = 0

    (l1, v1) = np.linalg.eigh(cov1)
    check_nonpositive_eigvals(l1)
    cov1_sqrt = (v1 * np.sqrt(l1)).dot(v1.T)
    cov_prod = cov1_sqrt.dot(cov2).dot(cov1_sqrt)
    lp = np.linalg.eigvalsh(cov_prod)
    check_nonpositive_eigvals(lp)

    trace = l1.sum() + np.trace(cov2) - 2 * np.sqrt(lp).sum()
    diff_mean = mean1 - mean2
    fd = diff_mean.dot(diff_mean) + trace

    return fd


class FrechetInceptionDistance(object):
    def __init__(self, generator, image_range=(-1, 1),
                 generator_postprocessing=None):

        self._inception_v3 = None
        self.generator = generator
        self.generator_postprocessing = generator_postprocessing
        self.image_range = image_range
        self._channels_axis = -1 if K.image_data_format() == "channels_last" else -3

    def _setup_inception_network(self):
        self._inception_v3 = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        self._pool_size = self._inception_v3.output_shape[-1]

    def _preprocess(self, images):
        if self.image_range != (-1, 1):
            images = images - self.image_range[0]
            images /= (self.image_range[1] - self.image_range[0]) / 2.0
            images -= 1.0
        if images.shape[self._channels_axis] == 1:
            images = np.concatenate([images] * 3, axis=self._channels_axis)
        return images

    def _stats(self, inputs, input_type="real", postprocessing=None,
               batch_size=64, num_batches=128, shuffle=True, seed=None):

        mean = np.zeros(self._pool_size)
        cov = np.zeros((self._pool_size, self._pool_size))
        N = 0

        if input_type == "generated":
            for i in tqdm.trange(len_dataset):
                batch = self.generator.predict(inputs(shape=((batch_size,).__add__(self.generator.input_shape[1:]))))
                batch = self._preprocess(batch)
                pool = self._inception_v3.predict(preprocess_image(batch), batch_size=batch_size)
                (mean, cov, N) = update_mean_cov(mean, cov, N, pool)
        else:
            for inputs in tqdm.tqdm(inputs, desc='Inner Epoch Loop', total=len_dataset):
                batch = self._preprocess(inputs)
                pool = self._inception_v3.predict(batch, batch_size=batch_size)
                (mean, cov, N) = update_mean_cov(mean, cov, N, pool)

        return (mean, cov)

    def __call__(self,
                 real_images,
                 batch_size=64,
                 num_batches_real=128,
                 num_batches_gen=None,
                 shuffle=True,
                 seed=None
                 ):

        if self._inception_v3 is None:
            self._setup_inception_network()
        # for inputs in tqdm(real_images, desc='Inner Epoch Loop', total=len_dataset):

        if num_batches_gen is None:
            num_batches_gen = num_batches_real
        (gen_mean, gen_cov) = self._stats(tf.random.normal,
                                          "generated", batch_size=batch_size, num_batches=num_batches_gen,
                                          postprocessing=self.generator_postprocessing,
                                          shuffle=shuffle, seed=seed)
        (real_mean, real_cov) = self._stats(real_images,
                                            "real", batch_size=batch_size, num_batches=num_batches_real,
                                            shuffle=shuffle, seed=seed)

        return frechet_distance(real_mean, real_cov, gen_mean, gen_cov)


def preprocess_image(img):
    img = tf.image.resize(img, [299, 299])
    return img

def make_mnist(batch_size, drop_remainder=True, shuffle=True, repeat=1):
    with np.load('mnist.npz', allow_pickle=True) as f:
        test_images, test_label = f['x_test'], f['y_test']

    test_images.shape = test_images.shape + (1,)

    @tf.function
    def _map_fn(img):
        img['data'] = tf.image.resize(img['data'], [299, 299])
        img['data'] = tf.clip_by_value(img['data'], 0, 255)
        img['data'] = img['data'] / 127.5-1
        return img['data']

    dataset = tl.memory_data_batch_dataset(test_images,
                                           test_label,
                                           batch_size,
                                           map_fn=_map_fn,
                                           shuffle=shuffle,
                                           repeat=repeat)
    img_shape = (32, 32, test_images.shape[-1])
    len_dataset = len(test_images) // batch_size

    return dataset, img_shape, len_dataset
import matplotlib.pyplot as plt
dataset, shape, len_dataset = make_mnist(64)

generator = tf.keras.models.load_model('G_Big.h5')

# real_images = preprocess_image(real_images[:10000])
# change (0,1) to the range of values in your dataset
fd = FrechetInceptionDistance(generator, (0, 1))
gan_fid = fd(dataset)
print(gan_fid)
