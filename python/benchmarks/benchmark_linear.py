import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate
from py_evpi import evpi, regression_evpi

from benchmark_problems import LinearBenchmarkProblem
plt.style.use("seaborn-v0_8-whitegrid")

p = LinearBenchmarkProblem()

nested_error = []
binning_error = []
regression_error = []
n_sample_range = range(1000, 50000, 1000)


def integral_evppi():
    # mathematical solution using intgration
    mu_y = p.COEFFICIENTS @ p.MU_X
    sigma_y = np.sqrt((p.COEFFICIENTS*p.COEFFICIENTS) @ (p.SIGMA_X*p.SIGMA_X))

    def emv_integrand(s):
        return scipy.stats.norm.pdf(s, mu_y, sigma_y) * s

    emv = np.max(scipy.integrate.quad_vec(emv_integrand, -500, 500)[0])

    evppis = np.zeros(3)
    for i in range(3):
        def inner(x_i):
            mask = np.ones(3, dtype=bool)
            mask[i] = False
            mu_i = p.COEFFICIENTS[:, mask] @ p.MU_X[mask] + \
                p.COEFFICIENTS[:, i] * x_i
            sigma_i = np.sqrt(
                (p.COEFFICIENTS[:, mask]*p.COEFFICIENTS[:, mask]) @ (p.SIGMA_X[mask]*p.SIGMA_X[mask]))

            def y_ev_pi(s):
                return scipy.stats.norm.pdf(s, mu_i, sigma_i) * s

            return np.max(scipy.integrate.quad_vec(y_ev_pi, -500, 500)[0])

        def ev_pi_integrand(x_i):
            return inner(x_i) * scipy.stats.norm.pdf(x_i, p.MU_X[i], p.SIGMA_X[i])
        outer = scipy.integrate.quad(ev_pi_integrand, -500, 500)[0]
        evppis[i] = outer-emv
    return (evppis)


def nested_mc_evppi(n_samples):
    n_samples_outer = int(np.sqrt(n_samples))
    n_samples_inner = int(np.sqrt(n_samples))

    evpis = np.zeros(3)
    for i in range(3):
        E_inner_vec = np.zeros(n_samples_outer)
        all_samples = np.zeros((n_samples_outer * n_samples_inner, 3))
        for outer_i in range(n_samples_outer):
            x_i = p.x_i(i)
            x = p.x(n_samples_inner)
            x[:, i] = x_i
            y = p.utility(x)
            all_samples[outer_i *
                        n_samples_inner:(outer_i+1)*n_samples_inner, :] = y
            E_inner_vec[outer_i] = np.max(np.mean(y, axis=0))
        emv = np.max(np.mean(all_samples, axis=0))
        evpis[i] = np.mean(E_inner_vec)-emv
    return evpis


# true_evppis = integral_evppi()
true_evppis = np.array(
    [19.015078040998887, 2.769151803744485, 9.6341106463947])

for j, N_SAMPLES in enumerate(n_sample_range):
    print(j)

    nested_mc_evppi_res = nested_mc_evppi(N_SAMPLES)

    x = p.x(N_SAMPLES)
    y = p.y()
    binning_evppi = evpi.multi_evppi(x, y)
    regression_evppi_res = regression_evpi.multi_evppi(x, y)

    def rms(diff):
        return (np.sqrt(np.sum(diff*diff)))

    nested_error.append(rms(true_evppis - nested_mc_evppi_res))
    binning_error.append(rms(true_evppis - binning_evppi))
    regression_error.append(rms(true_evppis - regression_evppi_res))

fig, ax = plt.subplots(1)
ax.plot(n_sample_range, nested_error, label="2-level nested MC")
ax.plot(n_sample_range, binning_error,
        label="1-level MC with binning")
ax.plot(n_sample_range, regression_error,
        label="1-level MC with regression")
ax.legend()
ax.set_xlabel("number of Monte Carlo samples")
ax.set_ylabel("root mean square error of the 3 EVPPI values")
ax.set_title(
    "benchmark on linear decision model\nwith 3 normally distributed parameters and 3 decision options")
plt.show()
