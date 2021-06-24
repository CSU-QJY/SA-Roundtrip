from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.exposure import rescale_intensity
from tensorflow import keras
import time
from sklearn import preprocessing


def GANmoduel(z, model):
    tf.keras.backend.clear_session()
    x_fake=tf.reshape(model(tf.constant(z,shape=(z.shape[0],100)),training=False),[-1,28,28])
    # x_fake=tf.reshape(model(tf.concat([tf.constant(z,shape=(z.shape[0],100)),np.eye(10)[np.random.randint(0,9,z.shape[0])]],axis=1),training=False),[-1,28,28])
    # x_fake = model(tf.constant(z, shape=(z.shape[0], 1, 1, 128), dtype=tf.float64), training=False)
    x = np.array(x_fake)
    # return (x.squeeze()+1)/2.0
    return x.squeeze()



if __name__ == "__main__":
    z1, z2, z3 = np.nditer(np.random.randn(100, 3), flags=['external_loop'], order='F')  # 128是已训练好模型的输入深度，必须为128
    z = np.row_stack((z1, z2, z3))
    start = time.time()
    model = keras.models.load_model('Roundtrip_model_G.h5')
    # model = keras.models.load_model('model.h5', compile=False)
    x1, x2, x3 = GANmoduel(z, model)
    print(time.time() - start)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(x1, cmap='gray')
    ax2.imshow(x2, cmap='gray')
    ax3.imshow(x3, cmap='gray')
    plt.show()

