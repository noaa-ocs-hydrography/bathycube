import numpy as np
from pytest import approx

from bathycube.numba_cube import run_cube_gridding as numbagrid
from bathycube.cube import run_cube_gridding as basegrid


def _build_data():
    _numpoints = 1000
    _x = np.random.uniform(low=403744.5, high=403747.5, size=_numpoints)
    _y = np.random.uniform(low=4122665.5, high=4122668.5, size=_numpoints)
    _z = np.random.uniform(low=13.0, high=15.0, size=_numpoints)
    _tvu = np.random.uniform(low=0.1, high=1.0, size=_numpoints)
    _thu = np.random.uniform(low=0.3, high=1.3, size=_numpoints)

    _z = _z.astype(np.float32)
    _tvu = _tvu.astype(np.float32)
    _thu = _thu.astype(np.float32)

    _numrows, _numcols = (3, 3)
    _resolution_x, _resolution_y = (1.0, 1.0)
    return [_z, _thu, _tvu, _x, _y, _numcols, _numrows, min(_x), max(_y), 'local', 'order1a', _resolution_x, _resolution_y]


def test_base():
    data_args = _build_data()

    ndata = numbagrid(*data_args)
    bdata = basegrid(*data_args)

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)


def test_posterior():
    data_args = _build_data()
    data_args[9] = 'posterior'

    ndata = numbagrid(*data_args)
    bdata = basegrid(*data_args)

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)


def test_prior():
    data_args = _build_data()
    data_args[9] = 'prior'

    ndata = numbagrid(*data_args)
    bdata = basegrid(*data_args)

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)


def test_predicted():
    data_args = _build_data()
    data_args[9] = 'predicted'

    ndata = numbagrid(*data_args)
    bdata = basegrid(*data_args)

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)


def test_base_cube():
    data_args = _build_data()

    ndata = numbagrid(*data_args, variance_selection='cube')
    bdata = basegrid(*data_args, variance_selection='cube')

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)


def test_base_input():
    data_args = _build_data()

    ndata = numbagrid(*data_args, variance_selection='input')
    bdata = basegrid(*data_args, variance_selection='input')

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)


def test_base_max():
    data_args = _build_data()

    ndata = numbagrid(*data_args, variance_selection='max')
    bdata = basegrid(*data_args, variance_selection='max')

    assert ndata[0] == approx(bdata[0], abs=0.001)
    assert ndata[1] == approx(bdata[1], abs=0.001)
    assert ndata[2] == approx(bdata[2], abs=0.001)
    assert ndata[3] == approx(bdata[3], abs=0.001)