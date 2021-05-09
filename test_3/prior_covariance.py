import numpy as np
import matplotlib.pyplot as plt

def prior_cov(height,width,gamma,d):
    """
    K(xy_i,xy_j) = gamma*exp(-||(x_i,y_i)-(x_j,y_j)||_2/d)
    """
    xs = np.linspace(0,1,width+1)
    delta_x = (xs[1]-xs[0])/2
    xs = xs[:-1]+delta_x

    ys = np.linspace(0,1,height+1)
    ys = ys[:-1]+delta_x

    [X,Y] = np.meshgrid(xs,ys)
    XYpoint = np.c_[X.flatten(),Y.flatten()]

    n = height*width
    dist_upper = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            dist_upper[i,j] = np.linalg.norm(XYpoint[i,:]-XYpoint[j,:])
    dist = dist_upper+dist_upper.T
    K = gamma*np.exp(-dist/d)
    return K


if __name__ == "__main__":
    height,width = 23,27
    gamma,d = 0.1,4
    K = prior_cov(height,width,gamma,d)
    plt.imshow(K)
    plt.colorbar()
    plt.show()