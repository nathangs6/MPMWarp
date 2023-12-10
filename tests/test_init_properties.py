import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import init_properties as src
import numpy as np
TOL = 0.00001


def test_init_cell_density():
    h = 0.1
    m_grid = np.array([1, 2, 3], dtype=np.float32)
    cell_density = np.empty(shape=(3), dtype=np.float32)
    src.init_cell_density(cell_density, m_grid, h)
    expected = [1000, 2000, 3000]
    for i in range(len(expected)):
        assert (cell_density[i] - expected[i]) <= TOL

def test_init_particle_weight():
    h = 0.1
    m_grid = np.array([1, 2], dtype=np.float32)
    wip = np.array([
        [1, 2, 3],
        [0.5, 1.3, 2.2]], dtype=np.float32)
    d = np.empty(shape=(3), dtype=np.float32)
    src.init_particle_density(d, m_grid, wip, h)
    expected = [2000, 4600, 7400]
    for i in range(len(expected)):
        assert (d[i] - expected[i]) <= TOL

def test_init_particle_volume():
    m = np.array([1, 2, 3], dtype=np.float32)
    d = np.array([0.2, 4.3, 2.5], dtype=np.float32)
    v = np.empty(shape=(3), dtype=np.float32)
    src.init_particle_volume(v, m, d)
    expected = [1/0.2, 2/4.3, 3/2.5]
    for i in range(len(expected)):
        assert (v[i] - expected[i]) <= TOL

