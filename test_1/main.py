import numpy as np
import h5py
import radon
import cv2
from util import prior_cov
from MCMC import MCMC
import matplotlib.pyplot as plt
height,width = 16,16
lenth=height*width
angleNum=23
A, irs, jcs = radon.mexFunction(angleNum, height, width)
image = h5py.File('pet.mat')
data = image['pet'][:]
data = np.transpose(data)
data = cv2.resize(data,(height,width))
u = data.reshape(data.shape[0] * data.shape[1], 1)
y_without_noise = A@u
snr = 0.1  # 信噪比
theta = snr * y_without_noise.max()
epsilon = np.random.randn(y_without_noise.shape[0],1)*theta
y = y_without_noise+epsilon

T = 10000
samples = np.zeros((T,lenth))
cov=np.eye(lenth)*0.0001
mean = np.random.randn(lenth,1)
gamma,d = 0.1,4
sigama1=prior_cov(angleNum,27,gamma,d)
sigama2=prior_cov(height,width,gamma,d)
MCMC(T,samples, mean, cov, y, sigama1, sigama2, A)
plt.imshow(data)
plt.colorbar()
plt.show()
