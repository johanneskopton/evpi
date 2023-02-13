import numpy as np
import pandas as pd

n_samples = int(1e5)

x = pd.DataFrame(columns=["x1", "x2", "x3"])
y = pd.DataFrame(columns=["y1", "y2", "y3"])

x.x1 = np.random.normal(10, 6, n_samples)
x.x2 = np.random.normal(-10, 2, n_samples)
x.x3 = np.random.normal(-20, 20, n_samples)

y.y1 = x.x1 + x.x2 + 2
y.y2 = -x.x2 + 0.5*x.x3
y.y3 = x.x1 * x.x2 - 5*x.x3

x.to_csv("x.csv")
y.to_csv("y.csv")
