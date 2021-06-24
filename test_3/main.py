import numpy as np
import cv2
from MCMC import MCMC
import matplotlib.pyplot as plt
from generator import GANmoduel
from tensorflow import keras
import myradon
from skimage.exposure import *
from skimage.metrics import peak_signal_noise_ratio
import tensorflow as tf
from ADMM_TV import TV
height, width = 28, 28
lenth = height * width
angleNum = 30
A= myradon.mexFunction(angleNum, height, width)
# A = np.eye(1024)
model=keras.models.load_model('Roundtrip_model_G.h5')
model_h=keras.models.load_model('Roundtrip_model_H.h5')
# model = keras.models.load_model('model.h5', compile=False)
# for i in range(2,10):
image_name = str(2) + '.jpg'
z_true = cv2.imread('mnist/' + image_name, cv2.IMREAD_GRAYSCALE) / 127.5-1
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

# z = np.random.randn(1, 100)
tv_image, tv_gamma, tv_psnr = TV(2, A, height, width, angleNum, snr)
z=np.array(model_h(tv_image))

z_mean, x_var, accept = MCMC(T, z, y, theta, A, model, step_size=0.01)

x_mean = rescale_intensity(GANmoduel(z_mean, model),out_range=(0.,1.))
x_true=rescale_intensity(x_true,out_range=(0.,1.))
psn = peak_signal_noise_ratio(x_true, x_mean)

# y_without_noise = np.transpose(y_without_noise.reshape(height, width))
y_without_noise = np.transpose(y_without_noise.reshape(angleNum, A.shape[0] // angleNum))


dx, dy = 0.5 * 180.0 / max(x_true.shape), 0.5 / y_without_noise.shape[1]
fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(16, 8))
ax1.set_title('true_image')
ax1.set_ylabel('MCMC')
ax2.set_title('MCMC_sample_image' + '\nPSNR=' + str(psn) + '\naccept=' + str(accept))
ax3.set_title('variance_image')
ax4.set_title('sinogram_image')
ax5.set_title('true_image')
ax5.set_ylabel('ADMM_TV')
ax6.set_title('tv_image')
ax6.set_xlabel(f'PSNR={tv_psnr} \nlambda={tv_gamma}')
ax7.set_title('Reconstruction error\nFiltered back projection')
ax8.set_title('sinogram_image')
ax1.imshow(x_true, cmap='gray')
ax2.imshow(x_mean, cmap='gray')
ax3.imshow(x_var.reshape(height, width), cmap='gray')
ax4.imshow(y_without_noise,
           extent=(-dx, 180.0 + dx, -dy, y_without_noise.shape[1] + dy),
           aspect='auto', cmap='gray')
ax5.imshow(x_true, cmap='gray')
ax6.imshow(tv_image, cmap='gray')
ax7.imshow(tv_image - x_true, cmap='gray')
ax8.imshow(y_without_noise,
           extent=(-dx, 180.0 + dx, -dy, y_without_noise.shape[1] + dy),
           aspect='auto', cmap='gray')
plt.suptitle(f'MCMC vs ADMM_TV\n   SNR={snr}')
plt.savefig('./MCMC_output/' + image_name)
plt.show()








