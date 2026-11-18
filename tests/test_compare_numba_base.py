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


def _build_step_data(n_first, n_second, depth_first, depth_second, vert_uncertainty):
    _numpoints = n_first + n_second
    _x = np.full(_numpoints, 1.5)
    _y = np.full(_numpoints, 1.5)
    _z = np.concatenate([np.full(n_first, depth_first),
                         np.full(n_second, depth_second)]).astype(np.float32)
    _tvu = np.full(_numpoints, vert_uncertainty, dtype=np.float32)
    _thu = np.full(_numpoints, 0.0001, dtype=np.float32)

    _numrows, _numcols = (3, 3)
    _resolution_x, _resolution_y = (1.0, 1.0)
    return [_z, _thu, _tvu, _x, _y, _numcols, _numrows, 0.0, 3.0, 'local', 'order1a', _resolution_x, _resolution_y]


def test_base_step_change():
    # all soundings land on one node, a run at one depth followed by a run at another.  quotient_limit is
    # passed to both so that the differing defaults cannot affect the comparison
    for args in [(100, 100, 15.0, 20.0, 0.01), (40, 40, 15.0, 18.0, 0.01),
                 (30, 30, 15.0, 16.0, 0.01), (25, 25, 15.0, 15.8, 0.005)]:
        data_args = _build_step_data(*args)

        ndata = numbagrid(*data_args, quotient_limit=255.0)
        bdata = basegrid(*data_args, quotient_limit=255.0)

        assert ndata[0][1][1] == approx(bdata[0][1][1], abs=0.001)
        assert ndata[1][1][1] == approx(bdata[1][1][1], abs=0.001)
