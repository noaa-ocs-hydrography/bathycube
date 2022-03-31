from pytest import approx

from bathycube.cube import *


def test_cube_params():
    param = CubeParameters()
    param.initialize('order1a', 0.5, 0.5)
    assert param.grid_resolution_x == 0.5
    assert param.grid_resolution_y == 0.5
    assert param.inv_dist_exponent == 1 / 2.0
    assert param.iho_order == 'order1a'


def test_cube_node_init():
    cb = CubeNode()
    cb.predicted_depth = 1.0
    assert cb.predicted_depth == 1.0
    assert cb.predicted_variance == 0.0


def test_cube_node_new_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 5.0)
    cb.add_hypothesis(6.0, 6.0)
    data = cb.hypotheses
    assert data[0].current_depth == 5.0
    assert data[1].current_depth == 6.0


def test_cube_node_remove_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 5.0)
    cb.remove_hypothesis(99.0)
    assert len(cb.hypotheses) == 1

    cb.remove_hypothesis(5.001)
    assert len(cb.hypotheses) == 0

    cb.add_hypothesis(5.0, 5.0)
    cb.add_hypothesis(6.0, 6.0)
    cb.add_hypothesis(7.0, 7.0)
    cb.add_hypothesis(8.0, 8.0)
    cb.remove_hypothesis(6.001)
    cb.remove_hypothesis(7.999)
    cb.remove_hypothesis(5.001)
    data = cb.hypotheses
    assert len(data) == 1
    assert data[0].current_depth == 7.0


def test_cube_node_nominate_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 5.0)
    cb.add_hypothesis(6.0, 6.0)
    cb.add_hypothesis(7.0, 7.0)
    cb.add_hypothesis(8.0, 8.0)

    assert cb.nominated is None
    cb.nominate_hypothesis(5.001)
    assert cb.nominated.current_depth == 5.0

    cb.nominate_hypothesis(4.999)
    assert cb.nominated.current_depth == 5.0

    cb.nominate_hypothesis(7.001)
    assert cb.nominated.current_depth == 7.0

    cb.nominate_hypothesis(7.5)
    assert cb.nominated is None


def test_cube_node_reset_nomination():
    cb = CubeNode()
    cb.clear_nomination()
    assert cb.nominated is None

    cb.add_hypothesis(8.0, 8.0)
    cb.nominate_hypothesis(8.001)
    cb.clear_nomination()
    assert cb.nominated is None


def test_cube_node_is_nominated():
    cb = CubeNode()
    assert not cb.has_nomination()
    assert cb.nominated is None

    cb.add_hypothesis(8.0, 8.0)
    cb.nominate_hypothesis(8.001)
    assert cb.has_nomination()


def test_cube_node_set_preddepth():
    cb = CubeNode()
    assert cb.predicted_depth == 0.0
    assert cb.predicted_variance == 0.0
    cb.predicted_depth = 5.0
    cb.predicted_variance = 1.5
    assert cb.predicted_depth == 5.0
    assert cb.predicted_variance == 1.5


def test_cube_node_monitor_hypothesis():
    cb = CubeNode()
    assert not cb.monitor_hypothesis(0, 1.0, 1.0)
    cb.add_hypothesis(5.0, 0.5)
    assert not cb.monitor_hypothesis(0, 10.0, 1.0)
    assert cb.monitor_hypothesis(0, 8.0, 1.0)  # no intervention required
    assert not cb.monitor_hypothesis(0, 8.0, 1.0)  # second monitor and the cum bayes fac is less than the threshold


def test_cube_node_reset_monitor():
    cb = CubeNode()
    assert not cb.reset_monitor(0)  # failed with no hypotheses
    cb.add_hypothesis(5.0, 0.5)
    assert cb.monitor_hypothesis(0, 8.0, 1.0)  # no intervention required
    hypo = cb.hypotheses[0]
    assert hypo.cum_bayes_fac == approx(0.166, abs=0.001)
    assert hypo.seq_length == 1.0
    cb.reset_monitor(0)
    assert hypo.cum_bayes_fac == 1.0
    assert hypo.seq_length == 0


def test_cube_node_update_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 1.0)
    cb.add_hypothesis(6.0, 1.0)
    cb.add_hypothesis(7.0, 1.0)

    hypo = cb.hypotheses[1]
    assert hypo.predict_depth == 6.0
    assert hypo.current_depth == 6.0
    assert hypo.current_variance == 1.0
    assert hypo.predict_variance == 1.0
    assert hypo.number_of_points == 1

    cb.update_hypothesis(1, 6.1, 0.9)
    assert hypo.predict_depth == approx(6.053, abs=0.001)
    assert hypo.current_depth == approx(6.053, abs=0.001)
    assert hypo.current_variance == approx(0.474, abs=0.001)
    assert hypo.predict_variance == approx(0.474, abs=0.001)
    assert hypo.number_of_points == 2


def test_cube_node_best_hypothesis_index():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 1.0)
    cb.add_hypothesis(6.0, 1.0)
    cb.add_hypothesis(7.0, 1.0)

    assert cb.best_hypothesis_index(5.4, 1.0) == 0
    assert cb.best_hypothesis_index(5.6, 1.0) == 1


def test_cube_node_update_node():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 1.0)
    cb.update_node(5.1, 1.0)

    assert len(cb.hypotheses) == 1
    assert cb.hypotheses[0].current_depth == approx(5.05, abs=0.01)
    assert cb.hypotheses[0].number_of_points == 2

    cb.update_node(15.1, 1.0)
    assert len(cb.hypotheses) == 2


def test_cube_node_truncate():
    cb = CubeNode()
    cb.add_to_queue(6.0, 1.0)
    cb.add_to_queue(6.1, 1.0)
    cb.add_to_queue(6.2, 1.0)
    cb.add_to_queue(6.3, 1.0)
    cb.add_to_queue(6.4, 1.0)
    cb.add_to_queue(16.5, 2.0)
    cb.add_to_queue(36.6, 2.0)
    assert cb.n_queued == 7
    cb.truncate()
    assert cb.n_queued == 6


def test_cube_node_queue_flush_node():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(5.1, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.3, 1.0)
    cb.n_queued = 4
    cb.flush_queue()

    hypos = cb.hypotheses
    assert len(hypos) == 1
    assert cb.n_queued == 0
    assert hypos[0].current_depth == approx(5.150, abs=0.001)
    assert hypos[0].current_variance == approx(0.25, abs=0.001)
    assert hypos[0].cum_bayes_fac == approx(1490.966, abs=0.001)
    assert hypos[0].number_of_points == 4


def test_cube_node_choose_hypothesis():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(5.1, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.n_queued = 5
    cb.flush_queue()

    hypo, ratio = cb.choose_hypothesis()
    assert hypo.number_of_points == 3
    assert ratio == 3.5


def test_cube_node_queue_fill():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.5, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.add_to_queue(4.5, 1.0)

    assert cb.n_queued == 6
    data = cb.queue
    assert np.allclose(data[0][0], np.array(4.5, dtype=np.float32))
    assert np.allclose(data[1][0], np.array(5.0, dtype=np.float32))
    assert np.allclose(data[2][0], np.array(5.2, dtype=np.float32))
    assert np.allclose(data[3][0], np.array(5.5, dtype=np.float32))
    assert np.allclose(data[4][0], np.array(17.7, dtype=np.float32))
    assert np.allclose(data[5][0], np.array(17.8, dtype=np.float32))


def test_cube_node_add_to_queue():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.5, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.add_to_queue(4.4, 1.0)
    cb.add_to_queue(4.0, 1.0)
    cb.add_to_queue(16.7, 1.0)
    cb.add_to_queue(4.2, 1.0)
    cb.add_to_queue(4.5, 1.0)
    cb.add_to_queue(16.8, 1.0)

    assert cb.hypotheses == []
    # this should trigger update node, as you hit median length limit
    cb.add_to_queue(4.6, 1.0)
    assert len(cb.hypotheses) == 1


def test_cube_node_queue_insert():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.5, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.add_to_queue(4.4, 1.0)
    cb.add_to_queue(4.0, 1.0)
    cb.add_to_queue(16.7, 1.0)
    cb.add_to_queue(4.2, 1.0)
    cb.add_to_queue(4.5, 1.0)
    cb.add_to_queue(16.8, 1.0)
    data = [d[0] for d in cb.queue]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cb.queue_insert(10.0, 1.0)
    assert np.allclose(np.array(5.2, dtype=np.float32), median_data[0])
    data = [d[0] for d in cb.queue]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.5, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cb.queue_insert(10.0, 1.0)
    assert np.allclose(np.array(5.5, dtype=np.float32), median_data[0])
    data = [d[0] for d in cb.queue]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cb.queue_insert(100.0, 1.0)  # this outlier will trigger truncation
    assert np.allclose(np.array(10.0, dtype=np.float32), median_data[0])
    data = [d[0] for d in cb.queue]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)


def test_cube_add_point_to_node():
    cb = CubeNode()  # predicted depth flagged
    cb.predicted_depth = np.float32(np.nan)
    cb.add_point_to_node(5.0, 0.0, 0.0, 1.0)
    assert cb.n_queued == 0

    cb = CubeNode()  # blunder
    cb.predicted_depth = 100.0
    cb.add_point_to_node(50.0, 0.0, 0.0, 1.0)
    assert cb.n_queued == 0

    cb = CubeNode()  # too far
    cb.add_point_to_node(5.0, 0.0, 0.0, 1.0)
    assert cb.n_queued == 0

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    assert cb.n_queued == 1

    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    assert cb.n_queued == 11
    assert cb.hypotheses == []

    dpths = [d[0] for d in cb.queue]
    assert np.allclose(dpths, np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)
    varis = [d[1] for d in cb.queue]
    assert np.allclose(varis, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), atol=0.01)


def test_cube_node_extract_depth_uncertainty():
    cb = CubeNode()
    d, u, r = cb.extract_node_value(('depth', 'uncertainty', 'ratio'))
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_node_value(('depth', 'uncertainty', 'ratio'))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    cb.nominate_hypothesis(5.0)
    d, u, r = cb.extract_node_value(('depth', 'uncertainty', 'ratio'))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_node_value(('depth', 'uncertainty', 'ratio'))
    assert d == approx(4.686, abs=0.001)
    assert u == approx(0.524, abs=0.001)
    assert r == approx(3.25, abs=0.001)


def test_cube_node_extract_closest_depth_uncertainty():
    cb = CubeNode()
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    cb.nominate_hypothesis(5.0)
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_extract_posterior_depth_uncertainty():
    cb = CubeNode()
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    cb.nominate_hypothesis(5.0)
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ('depth', 'uncertainty', 'ratio'))
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_hypothesis_count():
    cb = CubeNode()
    assert cb.return_number_of_hypotheses() == 0

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    assert cb.return_number_of_hypotheses() == 1

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    assert cb.return_number_of_hypotheses() == 2
