import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(train_images, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
indx=np.argwhere(train_label==2)

images=np.reshape(train_images[indx],[-1,784])
label= np.reshape(np.eye(10)[train_label][indx],[-1,10])
plt.imshow(np.reshape(images[656,:],[28,28]),cmap=plt.cm.Greys_r)
plt.savefig('2.jpg')


