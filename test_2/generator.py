from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
def GANmoduel(z):
    tf.keras.backend.clear_session()
    model = keras.models.load_model('model.h5',compile=False)
    z = tf.constant(z.reshape(1,-1)[np.newaxis,np.newaxis,:],dtype=tf.float32)
    x_fake = model(z)
    x = np.array(x_fake)
    x = x[0, :, :, 0]
    # img = im.immerge(x_fake, n_rows=10).squeeze()
    # plt.imshow(x)
    # plt.show()
    return x

if __name__=="__main__":
    z = np.random.randn(128,1) #128是已训练好模型的输入深度，必须为128
    start = time.time()
    x = GANmoduel(z)
    print(time.time()-start)
    plt.imshow(x,cmap='gray')
    plt.show()
