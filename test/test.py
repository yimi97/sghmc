import numpy as np
# import sys
# sys.path.append('../')
from sghmc import sghmc
from autograd import jacobian
import seaborn as sns
import matplotlib.pyplot as plt

# data and parameters
np.random.seed(663)
n = 100
x = np.zeros((100, 1))
theta_0 = np.array([0.0])
p = theta_0.shape[0]
eps = 0.1
C = np.eye(1)
V = np.eye(1)*4
size = 1
epochs = 4000
burns = 200

# gradient U
def gradU_noise(theta, x, n, size):
    '''noisy gradient from paper fig1'''
    return -4 * theta + 4 * theta**3 + np.random.normal(0, 2)

sim = sghmc(gradU_noise, eps, C, np.eye(p), theta_0, V, epochs, burns, x, size)

plt1 = sns.kdeplot(sim[0, :])
fig1 = plt1.get_figure()
fig1.savefig("example.png")



