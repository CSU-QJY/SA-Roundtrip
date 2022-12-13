import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def Rosenbrock(x, y):
    return np.exp(-8 * (y - x ** 2) ** 2 + (1 - x) ** 2)

x = np.random.uniform(-2, 4, 10000)
y = x**2 + np.random.randn(10000)*0.1
r = Rosenbrock(x, y)

sns.scatterplot(x=x,y=y)
plt.show()

sns.scatterplot(x=x, y=r)
plt.show()

