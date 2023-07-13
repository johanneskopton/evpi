import numpy as np
import pandas as pd
from py_evpi import evpi, evppi, multi_evppi, binary_evppi, evipi

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
    assert np.allclose(multi_evppi(x, y-100), [7.1, 2.3, 9.9], atol=atol)
    assert np.allclose(multi_evppi(x, y+100), [7.1, 2.3, 9.9], atol=atol)


def test_binary_evpi():
    assert np.isclose(binary_evppi(x.x1, y.y1), 5.9, atol=atol)


def test_evipi():
    a = []
    for i in range(1, 20, 2):
        a.append(evipi(x.x1, y, i))
    print(a)
    assert False
