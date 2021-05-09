import numpy as np
import radon
import cv2
from MCMC import MCMC
import matplotlib.pyplot as plt
from generator import GANmoduel
from tensorflow import keras
height,width = 32,32
lenth=height*width
angleNum=30
model = keras.models.load_model('model.h5', compile=False)
A, irs, jcs = radon.mexFunction(angleNum, height, width)
image_name='4.jpg'
z_true=cv2.imread('mnist/'+image_name,cv2.IMREAD_GRAYSCALE)/127.5-1
# z_true = np.random.randn(128,1).reshape(1,-1)
# x = GANmoduel(z_true)
x_true = cv2.resize(z_true,(height,width))
u = x_true.reshape(-1, 1)

y_without_noise = A@u
snr = 0.1  # 信噪比
theta = snr * abs(y_without_noise).max()
epsilon = np.random.randn(y_without_noise.shape[0],1)*theta
y = y_without_noise+epsilon
T = 50000
z = np.random.randn(128,1).reshape(1,-1)
z_mean= MCMC(T, z, y, theta, A,model,step_size=0.01)
x_mean = GANmoduel(z_mean,model)
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.imshow(x_true,cmap='gray')
ax2.imshow(x_mean,cmap='gray')
plt.show()


