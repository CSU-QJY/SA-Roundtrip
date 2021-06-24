from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
model_G = keras.models.load_model('Roundtrip_model_G.h5', compile=False)
test=model_G.predict(np.array(tf.random.normal(shape=(1, 110))))
plt.imshow(test.reshape([28,28]))
plt.show()
