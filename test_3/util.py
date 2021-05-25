import math
import numpy as np

def pyu(y, u, A,theta):
    p_yu = (y - A @ u).T @ (y - A @ u)/(theta**2)
    p_u = u.T @ u
    Phi = (p_yu + p_u) * (-0.5)
    return Phi

def psnr(img1, img2):
   mse = np.mean( (img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def prior_cov(height, width, gamma, d):
    """
    K(xy_i,xy_j) = gamma*exp(-||(x_i,y_i)-(x_j,y_j)||_2/d)
    """
    xs = np.linspace(0, 1, width + 1)
    delta_x = (xs[1] - xs[0]) / 2
    xs = xs[:-1] + delta_x

    ys = np.linspace(0, 1, height + 1)
    ys = ys[:-1] + delta_x

    [X, Y] = np.meshgrid(xs, ys)
    XYpoint = np.c_[X.flatten(), Y.flatten()]

    n = height * width
    dist_upper = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist_upper[i, j] = np.linalg.norm(XYpoint[i, :] - XYpoint[j, :])
    dist = dist_upper + dist_upper.T
    K = gamma * np.exp(-dist / d)
    return K


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def cauchy(x):
    y = ((math.e ** (-((x - 2) ** 2) / 0.5) / math.sqrt(2 * math.pi * 0.5)) + (
            math.e ** (-((x - 3) ** 2) / 0.18) / math.sqrt(2 * math.pi * 0.3))) / 2
    return y

# def pu(r, u_2):
#     I_2 = np.eye(u_2.shape[0])
#     p = math.e ** (-(u_2.T@np.linalg.inv(r * r * I_2)@u_2) / 2)
#     return p
