import random
import matplotlib.pyplot as plt
import numpy as np
import util


def MCMC(T, samples, mean, cov, y, sigama, theta, A):
    t = 0
    accept = 0
    sigama_t=0
    while t < T:
        if t % 100 == 0:
            print(t)
        t = t + 1
        theta_star = np.random.multivariate_normal(mean=mean.flatten(), cov=cov, size=(1)).T
        alpha = min(1, np.exp((util.pyu(y, util.sigmoid(theta_star), A, sigama, theta) - util.pyu(y, util.sigmoid(mean), A, sigama, theta))[0][0]))
        u = random.uniform(0, 1)
        if u <= alpha:
            mean = theta_star
            sigama_t=sigama_t+theta_star.flatten()
            accept += 1
        else:
            sigama_t=sigama_t+mean.flatten()
    print(f'accept rate: {accept / T * 100}')
    
    # a = np.sum(samples[:2, :] / (0 + 2), axis=0)
    plt.imshow((sigama_t/T).reshape(16, 16),vmin=-1,vmax=1)
    plt.colorbar()
    plt.show()
    return sigama_t
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
