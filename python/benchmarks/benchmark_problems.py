import numpy as np


class LinearBenchmarkProblem:
    def __init__(self):
        self.MU_X = np.array([4, 3, 6])
        self.SIGMA_X = np.array([8, 2, 15])
        self._x = None

    def utility(self, x):
        return (self.COEFFICIENTS @ x.T).T

    def x(self, N_SAMPLES):
        self._x = np.random.normal(self.MU_X, self.SIGMA_X, (N_SAMPLES, 3))
        return self._x

    def x_i(self, i):
        return np.random.normal(self.MU_X[i], self.SIGMA_X[i])

    def y(self):
        if self._x is None:
            raise ValueError("Cannot calculate y without x.")
        return self.utility(self._x)


class LinearBenchmarkProblem1(LinearBenchmarkProblem):
    def __init__(self):
        self.COEFFICIENTS = np.array([[-2, 3, 0],
                                      [5, -4, 0],
                                      [0, -3, 2]])
        super().__init__()


class LinearBenchmarkProblem2(LinearBenchmarkProblem):
    def __init__(self):
        self.COEFFICIENTS = np.array([[0, -20, 0],
                                      [0, 40, 5],
                                      [0, 8, 2]])
        super().__init__()


class NonlinearBenchmarkProblem3(LinearBenchmarkProblem):
    def __init__(self):
        self.COEFFICIENTS = np.array([[-2, 3, 0],
                                      [5, -4, 0],
                                      [0, -3, 2]])
        super().__init__()

    def utility(self, x):
        y = (self.COEFFICIENTS @ x.T).T
        y[:, 2] += 20
        return y


class NonlinearBenchmarkProblem(LinearBenchmarkProblem1):
    def utility(self, x):
        y = (self.COEFFICIENTS @ x.T).T
        y[x[:, 2] < -20, 2] -= 50
        y[x[:, 0] < -10, 0] += 150
        y[:, 1] *= y[:, 1] / 1000
        return y
