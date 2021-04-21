import numpy as np
import h5py
import radon
import cv2
from util import prior_cov
from MCMC import MCMC
import matplotlib.pyplot as plt
from te import GANmoduel
height,width = 16,16
lenth=height*width
angleNum=23
A, irs, jcs = radon.mexFunction(angleNum, height, width)
# image = h5py.File('pet.mat')
# data = image['pet'][:]
# data = np.transpose(data)
data = GANmoduel()
data = cv2.resize(data,(height,width))
u = data.reshape(data.shape[0] * data.shape[1], 1)
y_without_noise = A@u
snr = 0.1  # 信噪比
theta = snr * y_without_noise.max()
epsilon = np.random.randn(y_without_noise.shape[0],1)*theta
y = y_without_noise+epsilon

T = 50000
samples = np.zeros((T,lenth))
cov=np.eye(lenth)*0.001
mean = np.random.randn(lenth,1)
gamma,d = 0.2,4
sigama=prior_cov(height,width,gamma,d)
sigama_t=MCMC(T,samples, mean, cov, y, sigama, theta, A)
plt.imshow(data)
plt.colorbar()
plt.show()