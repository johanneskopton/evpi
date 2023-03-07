import numpy as np
import pandas as pd
from evpi import evpi, evppi, multi_evppi

x = pd.read_csv("../test_data/x.csv", index_col=0)
y = pd.read_csv("../test_data/y.csv", index_col=0)

# this is not a test for numerical precision but for general software testing
# so here a very generous tolerance is chosen to prevent random test fails due
# to unlucky Monte Carlo sampling
atol = 0.5


def test_evpi():
    assert np.isclose(evpi(y), 17.7, atol=atol)


def test_evppi():
    assert np.isclose(evppi(x.x1, y), 7.1, atol=atol)
    assert np.isclose(evppi(x.x2, y), 2.3, atol=atol)
    assert np.isclose(evppi(x.x3, y), 9.9, atol=atol)


def test_multi_evppi():
    assert np.allclose(multi_evppi(x, y), [7.1, 2.3, 9.9], atol=atol)


# def test_binary_evpi():
#     assert np.isclose(binary_evpi.binary_evppi(x.x1, y.y1), 5.9, atol=atol)
