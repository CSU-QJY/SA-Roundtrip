import random
import matplotlib.pyplot as plt
import numpy as np
import util


def MCMC(T, samples, mean, cov, y, theta, A):
    t = 0
    accept = 0
    while t < T:
        if t % 100 == 0:
            print(t)
        t = t + 1
        theta_star = np.random.multivariate_normal(mean=mean.flatten(), cov=cov, size=(1)).T
        alpha = min(1, np.exp(util.pyu(y, util.sigmoid(theta_star), theta, A) - util.pyu(y, util.sigmoid(mean), theta, A))[0][0])
        u = random.uniform(0, 1)
        if u <= alpha:
            mean = theta_star
            samples[[t-1], :] = theta_star.flatten()
            accept += 1
        else:
            samples[[t-1], :] = mean.flatten()
    print(f'accept rate: {accept / T * 100}')
    for i in range(samples.shape[0]):
        samples[[i],:]=np.sum(samples[:i+1,:]/(i+1),axis=0)
    # a = np.sum(samples[:2, :] / (0 + 2), axis=0)
    plt.imshow(samples.mean(axis=0).reshape(16, 16),vmin=0,vmax=1)
    plt.colorbar()
    plt.show()
    return samples
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
