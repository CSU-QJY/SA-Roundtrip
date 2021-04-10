import numpy as np
import h5py
import radon
import cv2
from MCMC import MCMC
import matplotlib.pyplot as plt
A, irs, jcs = radon.mexFunction(180, 16, 16)
image = h5py.File('pet.mat')
data = image['pet'][:]
data = np.transpose(data)
data = cv2.resize(data,(16,16))
u = data.reshape(data.shape[0] * data.shape[1], 1)
y_without_noise = A@u
snr = 0.1  # 信噪比
theta = snr * y_without_noise.max()
epsilon = np.random.randn(y_without_noise.shape[0],1)*theta
y = y_without_noise+epsilon

T = 10000
samples = np.zeros((T,256))
cov=np.eye(256)*0.0001
mean = np.random.randn(256,1)
MCMC(T,samples,mean,cov,y,theta,A)
plt.imshow(data)
plt.colorbar()
plt.show()