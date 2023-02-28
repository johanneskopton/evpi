import matplotlib.pyplot as plt

from benchmark_problems import LinearBenchmarkProblem

plt.style.use("seaborn-v0_8-whitegrid")


N_SAMPLES = 10000

nested_error = []
numerical_error = []

p = LinearBenchmarkProblem()

x = p.x(N_SAMPLES)
y = p.y()

fig, ax = plt.subplots(1, 2)
ax[0].hist(x, bins=100, histtype="stepfilled", alpha=0.5)
ax[1].hist(y, bins=100, histtype="stepfilled", alpha=0.5)
fig.tight_layout()
plt.show()


fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        ax[i][j].scatter(x[:, i], y[:, j], s=1)
fig.tight_layout()
plt.show()
