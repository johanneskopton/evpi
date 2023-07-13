from py_evpi import evipi, evppi
import benchmark_problems
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")


p = benchmark_problems.NonlinearBenchmarkProblem3()
n_samples = int(1e5)

x = p.x(n_samples)
y = p.y()


fig, axs = plt.subplots(1, 3, sharex=False, figsize=(10, 4))
for i in range(3):
    std = np.std(x[:, i])
    std_range = np.linspace(0.01, std, 10)
    evipis = np.array([evipi(x[:, i], y, std, n_bins=30)
                      for std in std_range])
    axs[i].plot(std_range, evipis)
    axs[i].set_title(str(evppi(x[:, i], y)))
    axs[i].hlines(evppi(x[:, i], y), min(std_range), max(std_range))
    axs[i].vlines(std, min(evipis), max(evipis))
fig.tight_layout()
plt.show()
