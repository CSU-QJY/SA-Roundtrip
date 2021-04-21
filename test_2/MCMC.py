import random
import matplotlib.pyplot as plt
import numpy as np
import util
from generator import GANmoduel

def MCMC(T, z, y, theta, A,step_size=0.01):
    t = 0
    accept = 0
    z_sum=np.zeros_like(z)
    while t < T:
        if t % 100 == 0:
            print(t)
        t = t + 1

        z_star = z + step_size*np.random.randn(*z.shape)         # w~N(0,I)
        x_star,x = GANmoduel(z_star).reshape(-1,1),GANmoduel(z).reshape(-1,1)
        alpha = min(1, np.exp((util.pyu(y, x_star, A, theta)
                             - util.pyu(y, x, A, theta))[0][0]))
        u = random.uniform(0, 1)
        if u <= alpha:
            z = z_star
            z_sum=z_sum+z_star
            accept += 1
        else:
            z_sum=z_sum+z
    print(f'accept rate: {accept / T * 100}')
    
    # a = np.sum(samples[:2, :] / (0 + 2), axis=0)
    # plt.imshow((z_sum/T).reshape(32, 32),vmin=-1,vmax=1)
    # plt.colorbar()
    # plt.show()
    return z_sum/T
# num_bins = 100
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)
# plt.sca(ax1)
# x = np.linspace(0,4,100)
# # plt.plot(x,cauchy(x))
# plt.sca(ax2)
# # plt.hist(samples, num_bins, density=True, stacked=True,facecolor='blue', alpha=0.5)
# plt.xlim([0,4])
# plt.show()
