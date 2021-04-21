from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
def GANmoduel():
    tf.keras.backend.clear_session()
    model = keras.models.load_model('model.h5')
    z = tf.random.normal((1, 1, 1, 128))
    x_fake = model(z)
    x = np.array(x_fake)
    x = x[0, :, :, 0]
    # img = im.immerge(x_fake, n_rows=10).squeeze()
    # plt.imshow(x)
    # plt.show()
    return x
