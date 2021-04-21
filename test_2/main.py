import numpy as np
import h5py
import radon
import cv2
from util import prior_cov
from MCMC import MCMC
import matplotlib.pyplot as plt
from generator import GANmoduel
import tensorflow as tf

height,width = 32,32
lenth=height*width
angleNum=23
A, irs, jcs = radon.mexFunction(angleNum, height, width)
# image = h5py.File('pet.mat')
# data = image['pet'][:]
# data = np.transpose(data)
z_true = np.random.randn(128,1)
x = GANmoduel(z_true)
# x = cv2.resize(x,(height,width))
u = x.reshape(-1, 1)

y_without_noise = A@u
snr = 0.1  # 信噪比
theta = snr * y_without_noise.max()
epsilon = np.random.randn(y_without_noise.shape[0],1)*theta
y = y_without_noise+epsilon

#%%
T = 500
z = np.random.randn(128,1)
z_mean= MCMC(T, z, y, theta, A,step_size=0.1)
#%%
x_mean = GANmoduel(z_mean)
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.imshow(x,cmap='gray')
ax2.imshow(x_mean,cmap='gray')
plt.show()


