import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate
from py_evpi import evpi
plt.style.use("seaborn-whitegrid")

COEFFICIENTS = np.array([[-2, 3, 0],
                         [5, -4, 0],
                         [0, -3, 2]])


def utility(x):
    y = (COEFFICIENTS @ x.T).T
    y[x[:, 2] < -20, 2] -= 50
    y[x[:, 0] < -10, 0] += 150
    y[:, 1] *= y[:, 1] / 1000
    return y


N_SAMPLES = 10000

nested_error = []
numerical_error = []

MU_X = np.array([4, 3, 6])
SIGMA_X = np.array([8, 2, 15])

x = np.random.normal(MU_X, SIGMA_X, (N_SAMPLES, 3))
y = utility(x)

fig, ax = plt.subplots(1, 2)
ax[0].hist(x, bins=100, histtype="stepfilled", alpha=0.5)
ax[1].hist(utility(x), bins=100, histtype="stepfilled", alpha=0.5)
fig.tight_layout()
plt.show()


fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        ax[i][j].scatter(x[:, i], y[:, j], s=1)
fig.tight_layout()
plt.show()
