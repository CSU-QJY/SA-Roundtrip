import numpy as np
import radon
import cv2
from MCMC import MCMC
import matplotlib.pyplot as plt
from generator import GANmoduel
from tensorflow import keras
from util import psnr

height, width = 32, 32
lenth = height * width
angleNum = 30
model = keras.models.load_model('model.h5', compile=False)
A= radon.mexFunction(angleNum, height, width)
for i in range(3,10):
    image_name = str(i) + '.jpg'
    z_true = cv2.imread('mnist/' + image_name, cv2.IMREAD_GRAYSCALE) / 255
    # z_true = np.random.randn(128,1).reshape(1,-1)
    # x = GANmoduel(z_true)
    x_true = cv2.resize(z_true, (height, width))
    u = x_true.reshape(-1, 1)

    y_without_noise = A @ u
    snr = 0.1  # 信噪比
    theta = snr * abs(y_without_noise).max()
    epsilon = np.random.randn(y_without_noise.shape[0], 1) * theta
    y = y_without_noise + epsilon
    T = 100000
    z = np.random.randn(128, 1).reshape(1, -1)
    z_mean, x_var, accept = MCMC(T, z, y, theta, A, model, step_size=0.01)
    x_mean = GANmoduel(z_mean, model)
    psn = psnr(x_true, x_mean)
    y_without_noise=np.transpose(y_without_noise.reshape(angleNum, A.shape[0] // angleNum))
    dx, dy = 0.5 * 180.0 / max(x_true.shape), 0.5 / y_without_noise.shape[1]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    ax1.set_title('true_image')
    ax2.set_title('sample_image')
    ax3.set_title('variance_image')
    ax4.set_title('sinogram_image')
    ax1.imshow(x_true, cmap='gray')
    ax2.imshow(x_mean, cmap='gray')
    ax3.imshow(x_var.reshape(height, width), cmap='gray')
    ax4.imshow(y_without_noise,
               extent=(-dx, 180.0 + dx, -dy, y_without_noise.shape[1] + dy),
               aspect='auto', cmap='gray')
    plt.suptitle('PSNR=' + str(psn) + '\naccept=' + str(accept))
    plt.savefig('./MCMC_output/' + str(i) + '.jpg')
    plt.show()
