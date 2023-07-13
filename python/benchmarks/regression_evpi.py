import numpy as np
# import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from scipy.stats import norm


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


def evipi(x, y, std):
    """Bullshit"""
    x = np.array(x)
    y = np.array(y)
    padding = 5*std
    minx = np.min(x)
    maxx = np.max(x)
    samples = np.linspace(minx-padding, maxx+padding, 20)
    bin_size = samples[1]-samples[0]
    # samples_x = np.linspace(minx, maxx, 20)

    yy = np.empty((len(samples), y.shape[1]))
    for i in range(y.shape[1]):
        gam = LinearGAM(s(0)).fit(x, y[:, i])
        yy[:, i] = gam.predict(samples)

    weighted_sum_outcome = 0
    for i, true_x in enumerate(samples):
        weighted_sum_outcome_bin = 0
        for j, obs_x in enumerate(samples):
            bin_prob = norm.cdf(obs_x+bin_size, true_x, std) - \
                norm.cdf(obs_x, true_x, std)
            selected_option_id = np.argmax(yy[j, :])
            outcome_bin = yy[i, selected_option_id]
            weighted_sum_outcome_bin += outcome_bin * bin_prob
        x_in_bin = (x > true_x) * (x < true_x + bin_size)
        bin_population = np.sum(x_in_bin)
        weighted_sum_outcome += weighted_sum_outcome_bin * bin_population
    ev_ipi = weighted_sum_outcome / len(x)

    yy = np.zeros(y.shape)
    for i in range(y.shape[1]):
        gam = LinearGAM(s(0)).fit(x, y[:, i])
        yy[:, i] = gam.predict(x)

    emv = np.max(np.mean(yy, axis=0))

    return ev_ipi - emv


def multi_evppi(x, y):
    x = np.array(x)
    y = np.array(y)
    evppis = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        evppis[i] = evppi(x[:, i], y)
    return evppis
