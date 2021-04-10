import math
import numpy as np

def pyu(y,u,theta,A):
    temp1 = (y-A@u).T@(y - A@u)/theta**2
    temp2 = u.T @ u/theta**2
    Phi = (temp1+temp2)*(-0.5)
    return Phi


def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def cauchy(x):
    y=((math.e ** (-((x - 2) ** 2) / 0.5) / math.sqrt(2 * math.pi * 0.5)) + (
            math.e ** (-((x - 3) ** 2) / 0.18) / math.sqrt(2 * math.pi * 0.3))) / 2
    return y

# def pu(r, u_2):
#     I_2 = np.eye(u_2.shape[0])
#     p = math.e ** (-(u_2.T@np.linalg.inv(r * r * I_2)@u_2) / 2)
#     return p
