import math
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cvxpy as cp
import numpy as np
import time
import cv2, os
import radon
from multiprocessing import Pool


def PSNR(ground_truth, predict):
    """
    计算单个图片的PSNR
    """
    ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())  # 归一化
    predict = (predict - predict.min()) / (predict.max() - predict.min())
    mse = np.mean((ground_truth - predict) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return np.round(psnr, 2)
# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    start = time.time()
    gamma.value = gamma_value
    result = p.solve(solver=cp.SCS)

    end = time.time()
    print(end - start)
    return x.value


height, width = 32, 32
lenth = height * width
angleNum = 30

i = 6
image_name = str(i) + '.jpg'
z_true = cv2.imread(os.path.join("mnist", image_name), cv2.IMREAD_GRAYSCALE) / 255

image = cv2.resize(z_true, (height, width))

height, width = image.shape
plt.imshow(image, cmap=plt.cm.Greys_r)
plt.colorbar()
plt.title(f"{height, width}")
signal_noise_ratio = 0.1

A = radon.mexFunction(angleNum, width, height)
y_clear = A @ image.flatten(order='F')
y_clear_2d = y_clear.reshape(-1, angleNum, order='F')

# add noise
noise_std = y_clear.max() * signal_noise_ratio
y_1d = y_clear + noise_std * np.random.randn(len(y_clear))
y_1d = np.clip(y_1d, 0, 100)
y = y_1d.reshape(-1, angleNum, order='F')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(y_clear_2d)
ax[0].set_title('clear data')

ax[1].imshow(y)
ax[1].set_title('noisy data')
# plt.show()
# Problem data.
n, m = A.shape
A = A
b = y_1d
gamma= cp.Parameter(nonneg=True)
x = cp.Variable(m)
f = cp.sum_squares(A @ x - b) / (2 * noise_std) + cp.multiply(gamma, cp.tv(x))

constraints = [0.0 <= x, x <= 1.0]
objective = cp.Minimize(f)
p = cp.Problem(objective, constraints)
gammas=np.linspace(1e-3, 10, 12).round(3).tolist()
print(gammas)
if __name__ == "__main__":

    # Parallel computation.

    pool = Pool(processes=12)
    par_x = pool.map(get_x, gammas)  # 已检查： x.value与gammas的顺序对应
    rec_imgs = [img.reshape(height, width, order='F') for img in par_x]
    psnrs = [PSNR(rec_img, image) for rec_img in rec_imgs]

    print(np.array(gammas)[np.array(psnrs).argmax()], np.array(psnrs).max())
    plt.figure(figsize=(20, 5))
    plt.plot(gammas, psnrs, 'ro-')
    plt.xlabel('lambda')
    plt.ylabel('PSNR')
    plt.grid()
    n_col = 5
    n_row = int(len(gammas) / n_col)
    _, axs = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 4))
    axs = axs.flatten()
    for img, ax, lambd, psnr in zip(rec_imgs, axs, gammas, psnrs):
        ax.imshow(img, cmap=plt.cm.Greys_r)
        ax.set_title(f"lambda={lambd} \nPSNR={psnr}")
    plt.show()