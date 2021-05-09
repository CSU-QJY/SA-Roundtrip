from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
def GANmoduel(z,model):
    tf.keras.backend.clear_session()
    x_fake = model(tf.constant(z,shape=(z.shape[0],1,1,128),dtype=tf.float64))
    x = np.array(x_fake)
    names=locals()
    sc=[]
    for i in range(z.shape[0]):
        names['x'+str(i+1)]=x[i,:,:,0]
        sc.append(names['x'+str(i+1)])
    sc=np.array(sc)
    return sc.squeeze()
    # x1,x2 = x[0, :, :, 0],x[1,:,:,0]
    # img = im.immerge(x_fake, n_rows=10).squeeze()
    # plt.imshow(x)
    # plt.show()


if __name__=="__main__":
    z1,z2,z3 = np.nditer(np.random.randn(128, 3), flags=['external_loop'], order='F')#128是已训练好模型的输入深度，必须为128
    z=np.row_stack((z1,z2,z3))
    start = time.time()
    model = keras.models.load_model('model.h5', compile=False)
    x1,x2,x3 = GANmoduel(z,model)
    print(time.time()-start)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(x1, cmap='gray')
    ax2.imshow(x2, cmap='gray')
    ax3.imshow(x3,cmap='gray')
    plt.show()

