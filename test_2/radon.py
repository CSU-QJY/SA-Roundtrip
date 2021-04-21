import math
import numpy as np


M_PI = math.pi


def MAXX(x, y):
    return x if x > y else y


def mexFunction(nt, nx, ny):
    xOrigin = int(MAXX(0, math.floor(nx / 2)))
    yOrigin = int(MAXX(0, math.floor(ny / 2)))
    Dr = 1
    Dx = 1
    rsize = math.ceil(math.sqrt(float(nx * nx + ny * ny) * Dx) / (2 * Dr)) + 1
    nr = 2 * rsize + 1
    xTable = np.zeros((1, nx))
    yTable = np.zeros((1, ny))
    yTable[0, 0] = (-yOrigin - 0.5) * Dx
    xTable[0, 0] = (-xOrigin - 0.5) * Dx
    for i in range(1, ny):
        yTable[0, i] = yTable[0, i - 1] + Dx
    for ii in range(1, nx):
        xTable[0, ii] = xTable[0, ii - 1] + Dx
    Dtheta = M_PI / nt
    percent_sparse = 2. / float(nr)
    nzmax = int(math.ceil(float(nr * nt * nx * ny * percent_sparse)))
    # nr=len(rho)
    # nt=len(theta)
    R = np.zeros((nr * nt, nx * ny))
    # R.resize(1,nr * nt*nx * ny)
    weight = np.zeros((1, nzmax))
    irs = np.zeros((1, nzmax))
    jcs = np.zeros((1, R.shape[1] + 1))
    k = 0
    for m in range(ny):
        for n in range(nx):
            jcs[0, m * nx + n] = k
            for j in range(nt):
                angle = j * Dtheta
                cosine = math.cos(angle)
                sine = math.sin(angle)
                xCos = yTable[0, m] * cosine + rsize * Dr
                ySin = xTable[0, n] * sine
                rldx = (xCos + ySin) / Dr
                rLow = math.floor(rldx)
                pixelLow = 1 - rldx + rLow
                if 0 <= rLow < (nr - 1):
                    irs[0, k] = nr * j + rLow  # irs为元素储存的行号
                    weight[0, k] = pixelLow
                    # R[int(irs[0,k]),int(jcs[0,m*nx+n])]=pixelLow
                    k = k + 1
                    irs[0, k] = nr * j + rLow + 1
                    weight[0, k] = 1 - pixelLow
                    # R[int(irs[0,k]),int(jcs[0,m*nx+n])]=1-pixelLow
                    k = k + 1
        jcs[0, nx * ny] = k
    for col in range(nx * ny):
        for row in range(2 * nt):
            R[int(irs[0, col * 2 * nt + row]), col] = weight[0, col * 2 * nt + row]

    return R, irs, jcs
# y=Au+E A为拉东变换矩阵，y为变换图像，u为原图像，E为噪声，已知y,A求u
# nb = A.shape[0]//64
# y_2d=Au.reshape(nb,64,order="F")
# image = rescale(data, scale=0.25, mode='reflect', multichannel=False)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
# ax1.set_title("Original")
# ax1.imshow(image)
# dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / y_2d.shape[0]
# ax2.set_title("Radon transform\n(Sinogram)")
# ax2.set_xlabel("Projection angle (deg)")
# ax2.set_ylabel("Projection position (pixels)")
# ax2.imshow(y_2d,
#            extent=(-dx, 180.0 + dx, -dy, y_2d.shape[0] + dy),
#            aspect='auto')
#
# fig.tight_layout()
# plt.show()
# R.reshape(1127,1024)
