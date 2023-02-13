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
MU_X = np.array([4, 3, 6])
SIGMA_X = np.array([8, 2, 15])


def utility(x):
    return (COEFFICIENTS @ x.T).T


nested_error = []
numerical_error = []
n_sample_range = range(1000, 100000, 1000)

for N_SAMPLES in n_sample_range:

    def integral_evpi():
        # mathematical solution using intgration
        mu_y = COEFFICIENTS @ MU_X
        sigma_y = np.sqrt((COEFFICIENTS*COEFFICIENTS) @ (SIGMA_X*SIGMA_X))

        def emv_integrand(s):
            return scipy.stats.norm.pdf(s, mu_y, sigma_y) * s

        emv = np.max(scipy.integrate.quad_vec(emv_integrand, -500, 500)[0])

        evpis = np.zeros(3)
        for i in range(3):
            def inner(x_i):
                mask = np.ones(3, dtype=bool)
                mask[i] = False
                mu_i = COEFFICIENTS[:, mask] @ MU_X[mask] + \
                    COEFFICIENTS[:, i] * x_i
                sigma_i = np.sqrt(
                    (COEFFICIENTS[:, mask]*COEFFICIENTS[:, mask]) @ (SIGMA_X[mask]*SIGMA_X[mask]))

                def y_ev_pi(s):
                    return scipy.stats.norm.pdf(s, mu_i, sigma_i) * s

                return np.max(scipy.integrate.quad_vec(y_ev_pi, -500, 500)[0])

            def ev_pi_integrand(x_i):
                return inner(x_i) * scipy.stats.norm.pdf(x_i, MU_X[i], SIGMA_X[i])
            outer = scipy.integrate.quad(ev_pi_integrand, -500, 500)[0]
            evpis[i] = outer-emv
        return (evpis)

    # true_evpi = integral_evpi()
    true_evpi = np.array(
        [19.015078040998887, 2.769151803744485, 9.6341106463947])

    def nested_mc_evpi():
        N_SAMPLES_OUTER = int(np.sqrt(N_SAMPLES))
        N_SAMPLES_INNER = int(np.sqrt(N_SAMPLES))

        evpis = np.zeros(3)
        for i in range(3):
            E_inner_vec = np.zeros(N_SAMPLES_OUTER)
            all_samples = np.zeros((N_SAMPLES_OUTER * N_SAMPLES_INNER, 3))
            for outer_i in range(N_SAMPLES_OUTER):
                x_i = np.random.normal(MU_X[i], SIGMA_X[i])
                x = np.random.normal(MU_X, SIGMA_X, (N_SAMPLES_INNER, 3))
                x[:, i] = x_i
                y = utility(x)
                all_samples[outer_i *
                            N_SAMPLES_INNER:(outer_i+1)*N_SAMPLES_INNER, :] = y
                E_inner_vec[outer_i] = np.max(np.mean(y, axis=0))
            emv = np.max(np.mean(all_samples, axis=0))
            evpis[i] = np.mean(E_inner_vec)-emv
        return evpis

    nested_mc_evpi_res = nested_mc_evpi()
    print(nested_mc_evpi_res)

    x = np.random.normal(MU_X, SIGMA_X, (N_SAMPLES, 3))
    y = utility(x)
    numerical_evpi = evpi.multi_evppi(x, y)

    def rms(diff):
        return (np.sqrt(np.sum(diff*diff)))

    nested_error.append(rms(true_evpi - nested_mc_evpi_res))
    numerical_error.append(rms(true_evpi - numerical_evpi))

fig, ax = plt.subplots(1)
ax.plot(n_sample_range, nested_error, label="2-level nested MC")
ax.plot(n_sample_range, numerical_error,
        label="1-level MC with binning")
ax.legend()
ax.set_xlabel("number of Monte Carlo samples")
ax.set_ylabel("root mean square error of the 3 EVPPI values")
ax.set_title(
    "benchmark on linear decision model\nwith 3 normally distributed parameters and 3 decision options")
plt.show()
