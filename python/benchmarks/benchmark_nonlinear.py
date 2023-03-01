import numpy as np
import matplotlib.pyplot as plt
import time
from py_evpi import evpi

import regression_evpi
from benchmark_problems import NonlinearBenchmarkProblem

plt.style.use("seaborn-v0_8-whitegrid")

p = NonlinearBenchmarkProblem()

nested_error = []
binning_error = []
regression_error = []

nested_time = 0
binning_time = 0
regression_time = 0

n_sample_range = range(1000, 50000, 1000)


def nested_mc_evppi(n_samples):
    n_samples_outer = int(np.sqrt(n_samples))
    n_samples_inner = int(np.sqrt(n_samples))

    evpis = np.zeros(3)
    for i in range(3):
        E_inner_vec = np.zeros(n_samples_outer, dtype=np.float32)
        all_samples_sum = np.zeros(3, dtype=np.float32)
        for outer_i in range(n_samples_outer):
            x_i = p.x_i(i)
            x = p.x(n_samples_inner)
            x[:, i] = x_i
            y = p.utility(x)
            all_samples_sum += np.sum(y, axis=0)
            E_inner_vec[outer_i] = np.max(np.mean(y, axis=0))
        emv = np.max(all_samples_sum / n_samples)
        evpis[i] = np.mean(E_inner_vec)-emv
    return evpis


# true_evppis = nested_mc_evppi(5e8)
true_evppis = np.array([[7.12395239, 2.31112576, 9.91058064]])

for j, N_SAMPLES in enumerate(n_sample_range):
    print(j)

    timer = time.time()
    nested_mc_evppi_res = nested_mc_evppi(N_SAMPLES)
    nested_time += time.time() - timer

    x = p.x(N_SAMPLES)
    y = p.y()

    timer = time.time()
    binning_evppi = evpi.multi_evppi(x, y)
    binning_time += time.time() - timer

    timer = time.time()
    regression_evppi_res = regression_evpi.multi_evppi(x, y)
    regression_time += time.time() - timer

    def rms(diff):
        return (np.sqrt(np.sum(diff*diff)))

    nested_error.append(rms(true_evppis - nested_mc_evppi_res))
    binning_error.append(rms(true_evppis - binning_evppi))
    regression_error.append(rms(true_evppis - regression_evppi_res))

fig, ax = plt.subplots(1)
ax.plot(n_sample_range, nested_error,
        label="2-level nested MC ({:.2f} s)".format(nested_time))
ax.plot(n_sample_range, binning_error,
        label="1-level MC with binning ({:.2f} s)".format(binning_time))
ax.plot(n_sample_range, regression_error,
        label="1-level MC with regression ({:.2f} s)".format(regression_time))
ax.legend()
ax.set_xlabel("number of Monte Carlo samples")
ax.set_ylabel("root mean square error of the 3 EVPPI values")
ax.set_title(
    "benchmark on nonlinear decision model\nwith 3 normally distributed parameters and 3 decision options")
plt.show()
