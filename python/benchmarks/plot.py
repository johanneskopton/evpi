import matplotlib.pyplot as plt

from benchmark_problems import LinearBenchmarkProblem1,\
    LinearBenchmarkProblem2, NonlinearBenchmarkProblem

plt.style.use("seaborn-v0_8-whitegrid")


N_SAMPLES = 10000

problems = [LinearBenchmarkProblem1,
            LinearBenchmarkProblem2,
            NonlinearBenchmarkProblem,
            ]

for Problem in problems:
    p = Problem()

    x = p.x(N_SAMPLES)
    y = p.y()

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(x, bins=100, histtype="stepfilled", alpha=0.5)
    ax[1].hist(y, bins=100, histtype="stepfilled", alpha=0.5)
    fig.tight_layout()

    fig, ax = plt.subplots(3, 3, figsize=(6, 6), sharex="col", sharey="row")
    for i in range(3):
        for j in range(3):
            if p.COEFFICIENTS[j, i] == 0 and not \
                    Problem == NonlinearBenchmarkProblem:
                color = "lightblue"
            else:
                color = "C0"
            ax[j][i].scatter(x[:, i], y[:, j], s=1, c=color)
        ax[2][i].set_xlabel("input {}".format(i))
    for j in range(3):
        ax[j][0].set_ylabel("option {}".format(j))

    fig.tight_layout()
    plt.show()
