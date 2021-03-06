import sghmc
import numpy as np

np.random.seed(663)
n = 100
x = np.zeros((100, 1))
theta_0 = np.array([0.0])
p = theta_0.shape[0]
eps = 0.1
C = np.eye(1)
V = np.eye(1)*4
size = 1
epochs = 400
burns = 200

def gradU_noise(theta, x, n, size):
    '''noisy gradient from paper fig1'''
    return -4 * theta + 4 * theta**3 + np.random.normal(0, 2)

sim = sghmc.sghmc(gradU_noise, eps, C, np.eye(p), theta_0, V, epochs, burns, x, size)

print("TEST")