import numpy as np
# import matplotlib.pyplot as plt
from pygam import LinearGAM, s


def evppi(x, y):
    x = np.array(x)
    y = np.array(y)
    yy = np.zeros(y.shape)
    for i in range(y.shape[1]):
        gam = LinearGAM(s(0)).fit(x, y[:, i])
        yy[:, i] = gam.predict(x)

    # plt.scatter(x, y[:, 0])
    # plt.scatter(x, yy[:, 0])
    # plt.show()
    evpi = np.mean(np.max(yy, axis=1)) - np.max(np.mean(yy, axis=0))
    return evpi


def multi_evppi(x, y):
    x = np.array(x)
    y = np.array(y)
    evppis = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        evppis[i] = evppi(x[:, i], y)
    return evppis
