import numpy as np
import matplotlib.pyplot as plt
import time

import py_evpi
import sorting_evpi
from benchmark_problems import LinearBenchmarkProblem1

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "text.usetex": True,
})

p = LinearBenchmarkProblem1()

n_samples = 10**5
n_bins = int(n_samples ** (1/3))


fig, ax = plt.subplots(1, figsize=(5, 5))


hist_evppis = []
sorting_evppis = []

hist_time = 0
sorting_time = 0

bins = list(range(1, 101, 3))
for i in bins:
    time_start = time.time()
    hist_evppis.append(py_evpi.multi_evppi(
        p.x(n_samples), p.y(), i))
    hist_time += time.time()-time_start

    time_start = time.time()
    sorting_evppis.append(sorting_evpi.sorting_multi_evppi(
        p.x(n_samples), p.y(), i))
    sorting_time += time.time()-time_start

hist_evppis = np.array(hist_evppis)
sorting_evppis = np.array(sorting_evppis)
for i in range(3):
    ax.plot(bins, hist_evppis[:, i],
            label="histogram EVPPI (variable {}, {} s)".format(i, hist_time),
            c="C"+str(i))
    ax.plot(bins, sorting_evppis[:, i],
            label="sorting EVPPI (variable {}, {} s)".format(i, sorting_time),
            c="C"+str(i),
            linestyle="dotted")
ax.legend()

fig.suptitle("{} samples".format(n_samples))
fig.tight_layout()
plt.show()
