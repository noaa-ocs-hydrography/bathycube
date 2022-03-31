from pytest import approx

from bathycube.numba_cube import *


def test_hypothesislist():
    ll = HypothesisList(return_new_hypothesis(5.0, 5.0), None)
    ll = ll.prepend(return_new_hypothesis(4.0, 4.0))
    ll = ll.prepend(return_new_hypothesis(3.0, 3.0))
    ll.append(return_new_hypothesis(7.0, 7.0))
    ll.append(return_new_hypothesis(8.0, 8.0))
    ll.insert(return_new_hypothesis(6.0, 6.0), 3)
    ll.insert(return_new_hypothesis(99.0, 99.0), 2)
    badindex = ll.get_nearest_in_depth(95.7, 10.0)
    ll.remove(badindex)
    ll = ll.drop_first()
    
    data = [d.current_depth for d in ll.get_data()]
    assert data == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert ll.get_item(0).current_depth == 4.0
    assert ll.get_item(2).current_depth == 6.0
    assert ll.get_item(4).current_depth == 8.0
    assert ll.get_nearest_min_error(5.5, 5.5) == 2
    assert ll.get_nearest_min_error(5.4, 5.5) == 1

    ll.get_item(2).number_of_samples = 3
    ll.get_item(1).number_of_samples = 2
    idx, curmax, secmax = ll.get_max_sample()
    assert idx == 2
    assert curmax == 3
    assert secmax == 2


def test_queuelist():
    ll = QueueList(return_new_sounding(5.0, 5.0, 0.0, 0.0), None)
    ll = ll.prepend(return_new_sounding(4.0, 4.0, 0.0, 0.0))
    ll.append(return_new_sounding(6.0, 6.0, 0.0, 0.0))
    ll.append(return_new_sounding(8.0, 8.0, 0.0, 0.0))
    ll.insert(return_new_sounding(7.0, 7.0, 0.0, 0.0), 3)
    ll.insert(return_new_sounding(99.0, 99.0, 0.0, 0.0), 2)
    ll.remove(2)
    ll = ll.drop_first()
    data = [d.depth for d in ll.get_data()]
    assert data == [5.0, 6.0, 7.0, 8.0]
    assert ll.get_item(0).depth == 5.0
    assert ll.get_item(2).depth == 7.0
    assert ll.get_item(3).depth == 8.0


def test_cube_params():
    param = return_default_cube_parameters('order1a', 0.5, 0.5)
    assert param.grid_resolution_x == 0.5
    assert param.grid_resolution_y == 0.5
    assert param.inv_dist_exponent == 1 / 2.0
    assert param.iho_order == 'order1a'


def test_cube_node_init():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(0.0, 0.0, 0.0, 0.0), None)
    cb.predicted_depth = 1.0
    assert cb.predicted_depth == 1.0
    assert cb.predicted_variance == 0.0
    assert np.array_equal(cb.queue.data.depth, np.array(0.0))


def test_cube_node_new_hypothesis():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 6.0, 0.0, 0.0))
    data = cb.hypotheses.get_data()
    assert data[0].current_depth == 5.0
    assert data[1].current_depth == 6.0


def test_cube_node_remove_hypothesis():
    cb = return_new_cubenode()
    assert not cube_node_remove_hypothesis(cb, 5.0)
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    assert not cube_node_remove_hypothesis(cb, 99.0)
    assert cube_node_remove_hypothesis(cb, 5.001)
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 6.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 7.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube_node_remove_hypothesis(cb, 6.001)
    assert cube_node_remove_hypothesis(cb, 7.999)
    assert cube_node_remove_hypothesis(cb, 5.001)
    data = cb.hypotheses.get_data()
    assert len(data) == 1
    assert data[0].current_depth == 7.0


def test_cube_node_nominate_hypothesis():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 6.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 7.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cb.nominated is None
    assert cube_node_nominate_hypothesis(cb, 5.001)
    assert cb.nominated.current_depth == 5.0
    assert cube_node_nominate_hypothesis(cb, 4.999)
    assert cb.nominated.current_depth == 5.0
    assert cube_node_nominate_hypothesis(cb, 7.001)
    assert cb.nominated.current_depth == 7.0
    assert not cube_node_nominate_hypothesis(cb, 7.5)
    assert cb.nominated is None


def test_cube_node_reset_nomination():
    cb = return_new_cubenode()
    assert cube_node_reset_nomination(cb)
    assert cb.nominated is None
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube_node_nominate_hypothesis(cb, 8.001)
    assert cube_node_reset_nomination(cb)
    assert cb.nominated is None


def test_cube_node_is_nominated():
    cb = return_new_cubenode()
    assert not cube_node_is_nominated(cb)
    assert cb.nominated is None
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube_node_nominate_hypothesis(cb, 8.001)
    assert cube_node_is_nominated(cb)


def test_cube_node_set_preddepth():
    cb = return_new_cubenode()
    assert cb.predicted_depth == 0.0
    assert cb.predicted_variance == 0.0
    cube_node_set_preddepth(cb, return_new_sounding(5.0, 1.5, 0.0, 0.0))
    assert cb.predicted_depth == 5.0
    assert cb.predicted_variance == 1.5


def test_cube_node_monitor_hypothesis():
    cb = return_new_cubenode()
    assert not cube_node_monitor_hypothesis(cb, 0, return_new_sounding(1.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 0.5, 0.0, 0.0))
    assert not cube_node_monitor_hypothesis(cb, 0, return_new_sounding(10.0, 1.0, 0.0, 0.0))  # trigger bayes factor less than minimum threshold
    assert cube_node_monitor_hypothesis(cb, 0, return_new_sounding(8.0, 1.0, 0.0, 0.0))  # no intervention required
    assert not cube_node_monitor_hypothesis(cb, 0, return_new_sounding(8.0, 1.0, 0.0, 0.0))  # second monitor and the cum bayes fac is less than the threshold


def test_cube_node_reset_monitor():
    cb = return_new_cubenode()
    assert not cube_node_reset_monitor(cb, 0)  # failed with no hypotheses
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 0.5, 0.0, 0.0))
    assert cube_node_monitor_hypothesis(cb, 0, return_new_sounding(8.0, 1.0, 0.0, 0.0))  # no intervention required
    hypo = cb.hypotheses.get_item(0)
    assert hypo.cum_bayes_fac == approx(0.166, abs=0.001)
    assert hypo.seq_length == 1.0
    cube_node_reset_monitor(cb, 0)
    assert hypo.cum_bayes_fac == 1.0
    assert hypo.seq_length == 0


def test_cube_node_update_hypothesis():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 1.0, 0.0, 0.0))

    hypo = cb.hypotheses.get_item(1)
    assert hypo.predict_depth == 6.0
    assert hypo.current_depth == 6.0
    assert hypo.current_variance == 1.0
    assert hypo.predict_variance == 1.0
    assert hypo.number_of_samples == 1

    cube_node_update_hypothesis(cb, 1, return_new_sounding(6.1, 0.9, 0.0, 0.0))
    assert hypo.predict_depth == approx(6.053, abs=0.001)
    assert hypo.current_depth == approx(6.053, abs=0.001)
    assert hypo.current_variance == approx(0.474, abs=0.001)
    assert hypo.predict_variance == approx(0.474, abs=0.001)
    assert hypo.number_of_samples == 2


def test_cube_node_best_hypothesis_index():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 1.0, 0.0, 0.0))
    assert cube_node_best_hypothesis_index(cb, return_new_sounding(5.4, 1.0, 0.0, 0.0)) == 0
    assert cube_node_best_hypothesis_index(cb, return_new_sounding(5.6, 1.0, 0.0, 0.0)) == 1


def test_cube_node_update_node():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_update_node(cb, return_new_sounding(5.1, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 1
    assert cb.hypotheses.get_item(0).current_depth == approx(5.05, abs=0.01)
    assert cb.hypotheses.get_item(0).number_of_samples == 2

    cube_node_update_node(cb, return_new_sounding(15.1, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 2


def test_cube_node_truncate():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(6.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(return_new_sounding(6.1, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(6.2, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(6.3, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(6.4, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(16.5, 2.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(36.6, 2.0, 0.0, 0.0))
    cb.n_queued = 7
    cube_node_truncate(cb)
    assert cb.n_queued == 6


def test_cube_node_queue_flush_node():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(5.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(return_new_sounding(5.1, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(5.3, 1.0, 0.0, 0.0))
    cb.n_queued = 4
    cube_node_queue_flush_node(cb)

    hypos = cb.hypotheses.get_data()
    assert len(hypos) == 1
    assert cb.n_queued == 0
    assert hypos[0].current_depth == approx(5.150, abs=0.001)
    assert hypos[0].current_variance == approx(0.25, abs=0.001)
    assert hypos[0].cum_bayes_fac == approx(1490.964, abs=0.001)
    assert hypos[0].number_of_samples == 4


def test_cube_node_choose_hypothesis():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(5.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(return_new_sounding(5.1, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cb.n_queued = 5
    cube_node_queue_flush_node(cb)
    hypo, ratio = cube_node_choose_hypothesis(cb)
    assert hypo.number_of_samples == 3
    assert ratio == 3.5


def test_cube_node_queue_fill():
    cb = return_new_cubenode()
    cube_node_queue_fill(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.5, 1.0, 0.0, 0.0))
    assert cb.n_queued == 6
    data = cb.queue.get_data()
    assert np.array_equal(data[0].depth, np.array(4.5, dtype=np.float32))
    assert np.array_equal(data[1].depth, np.array(5.0, dtype=np.float32))
    assert np.array_equal(data[2].depth, np.array(5.2, dtype=np.float32))
    assert np.array_equal(data[3].depth, np.array(5.5, dtype=np.float32))
    assert np.array_equal(data[4].depth, np.array(17.7, dtype=np.float32))
    assert np.array_equal(data[5].depth, np.array(17.8, dtype=np.float32))


def test_cube_node_add_to_queue():
    cb = return_new_cubenode()
    cube_node_add_to_queue(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.4, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.0, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(16.7, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.2, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.5, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(16.8, 1.0, 0.0, 0.0))
    assert cb.hypotheses is None
    # this should trigger update node, as you hit median length limit
    cube_node_add_to_queue(cb, return_new_sounding(4.6, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 1


def test_cube_node_queue_insert():
    cb = return_new_cubenode()
    cube_node_queue_fill(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.4, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.0, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(16.7, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.2, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.5, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(16.8, 1.0, 0.0, 0.0))
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube_node_queue_insert(cb, return_new_sounding(10.0, 1.0, 0.0, 0.0))
    assert np.allclose(np.array(5.2, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.5, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube_node_queue_insert(cb, return_new_sounding(10.0, 1.0, 0.0, 0.0))
    assert np.allclose(np.array(5.5, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube_node_queue_insert(cb, return_new_sounding(100.0, 1.0, 0.0, 0.0))  # this outlier will trigger truncation
    assert np.allclose(np.array(10.0, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)


def test_cube_node_insert():
    cb = return_new_cubenode()  # predicted depth flagged
    cb.predicted_depth = np.float32(np.nan)
    handled = cube_node_insert(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = return_new_cubenode()  # blunder
    cb.predicted_depth = 100.0
    handled = cube_node_insert(cb, return_new_sounding(50.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = return_new_cubenode()  # too far
    handled = cube_node_insert(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = return_new_cubenode()
    handled = cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    assert handled
    assert cb.n_queued == 1

    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    assert cb.n_queued == 11
    assert cb.hypotheses is None

    dpths = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(dpths, np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)
    varis = [d.variance for d in cb.queue.get_data()]
    assert np.allclose(varis, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), atol=0.01)


def test_cube_node_extract_depth_uncertainty():
    cb = return_new_cubenode()
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert d == approx(4.686, abs=0.001)
    assert u == approx(0.524, abs=0.001)
    assert r == approx(3.25, abs=0.001)


def test_cube_node_extract_closest_depth_uncertainty():
    cb = return_new_cubenode()
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_extract_posterior_depth_uncertainty():
    cb = return_new_cubenode()
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_hypothesis_count():
    cb = return_new_cubenode()
    assert cube_node_hypothesis_count(cb) == 0

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    assert cube_node_hypothesis_count(cb) == 1

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    assert cube_node_hypothesis_count(cb) == 2
