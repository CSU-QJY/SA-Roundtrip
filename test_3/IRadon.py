import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage.transform import radon, rescale, iradon
import cv2
from myradon import mexFunction
from util import psnr
for i in range(10):
    image_name = str(i) + '.jpg'
    z_true = cv2.imread('mnist/' + image_name, cv2.IMREAD_GRAYSCALE)/255
    image = rescale(z_true, scale=1, mode='reflect', multichannel=False)
    height, width = 28, 28
    theta = np.linspace(0., 180., 30, endpoint=False)
    sinogram, R = radon(image, theta=theta, circle=True)
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
    error = reconstruction_fbp - image
    print(f"FBP rms reconstruction error: {np.sqrt(np.mean(error ** 2)):.3g}")
    psn = psnr(image, reconstruction_fbp)
    print('psnr=' + str(psn))
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    ax1.set_title("Original")
    ax1.imshow(image, cmap='gray')
    ax2.set_title("Reconstruction\nFiltered back projection")
    ax2.imshow(reconstruction_fbp, cmap='gray')
    ax3.set_title("Reconstruction error\nFiltered back projection")
    ax3.imshow(reconstruction_fbp - image, cmap='gray', **imkwargs)
    ax4.set_title("Radon transform\n(Sinogram)")
    ax4.set_xlabel("Projection angle (deg)")
    ax4.set_ylabel("Projection position (pixels)")
    ax4.imshow(sinogram,
               extent=(-dx, 180.0 + dx, -dy, 30 + dy),
               aspect='auto', cmap='gray')

    fig.tight_layout()
    plt.suptitle('PSNR=' + str(psn))
    plt.savefig('./IRadon_output/' + str(i) + '.jpg')
    plt.show()
