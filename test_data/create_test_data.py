import numpy as np
import pandas as pd

n_samples = int(1e5)


class TestProblem:
    def __init__(self):
        self.COEFFICIENTS = np.array([[-2, 3, 0],
                                      [5, -4, 0],
                                      [0, -3, 2]])
        self.MU_X = np.array([4, 3, 6])
        self.SIGMA_X = np.array([8, 2, 15])
        self._x = None

    def utility(self, x):
        y = (self.COEFFICIENTS @ x.T).T
        y[x[:, 2] < -20, 2] -= 50
        y[x[:, 0] < -10, 0] += 150
        y[:, 1] *= y[:, 1] / 1000
        return y

    def x(self, N_SAMPLES):
        self._x = np.random.normal(self.MU_X, self.SIGMA_X, (N_SAMPLES, 3))
        return self._x

    def y(self):
        if self._x is None:
            raise ValueError("Cannot calculate y without x.")
        return self.utility(self._x)


p = TestProblem()

x = pd.DataFrame(columns=["x1", "x2", "x3"])
y = pd.DataFrame(columns=["y1", "y2", "y3"])

x[["x1", "x2", "x3"]] = p.x(n_samples)
y[["y1", "y2", "y3"]] = p.y()

x.to_csv("x.csv")
y.to_csv("y.csv")
