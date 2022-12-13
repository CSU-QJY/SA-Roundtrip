
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
with np.load('mnist.npz', allow_pickle=True) as f:
    x, train_label = f['x_train'], f['y_train']


# height, width = 256, 256
fbp=[]
sin_noise=[]
for i in tqdm.trange(x.shape[0]):
    theta = np.linspace(0., 180., 16, endpoint=False)
    sinogram = radon(x[i], theta=theta, circle=False)
    sinogram_noise = sinogram + np.random.normal(size=[sinogram.shape[0], sinogram.shape[1]]) * 0.01 * abs(
        sinogram).max()
    reconstruction_fbp = iradon(sinogram_noise, theta=theta, circle=False).astype(np.float32)
    sin_noise.append(sinogram_noise)
    fbp.append(reconstruction_fbp)
data={'x':x,'y':sin_noise,'fbp':fbp,'label':train_label}