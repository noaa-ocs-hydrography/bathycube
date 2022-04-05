import sys, os
import numpy as np
import json
from enum import Enum

# ex using jitclass with composition: https://stackoverflow.com/questions/38682260/how-to-nest-numba-jitclass
from numba import types, typed, deferred_type
from numba.core.types import unicode_type as numbastr
from numba import float64 as numbaf64
from numba import float32 as numbaf32
from numba import int64 as numbai64
from numba import int32 as numbai32
from numba import boolean as numbabool
from numba.experimental import jitclass
from numba import jit, njit, config, typeof, optional, typed


Debug = False


@njit
def get_iho_limits(iho_order: str):
    """
    Get fixed and variable Total Vertical Uncertainty components for the different IHO Order categories, see S-44
    Table 1 - Minimum Bathymetry Standards for Safety of Navigation Hydrographic Surveys

    Parameters
    ----------
    iho_order
        string representation of one of the IHO order categories, i.e. 'special' or 'order1a'

    Returns
    -------
    float
        'a' component, the fixed component of the TVU equation
    float
        'b' component, the variable component of the TVU equation
    """
    if iho_order == 'exclusive':
        return 0.15, 0.0075
    elif iho_order == 'special':
        return 0.25, 0.0075
    elif iho_order == 'order1a':
        return 0.5, 0.013
    elif iho_order == 'order1b':
        return 0.5, 0.013
    elif iho_order == 'order2':
        return 1.0, 0.023


cube_params_spec = {'iho_order': numbastr, 'grid_resolution_x': numbaf32, 'grid_resolution_y': numbaf32,
                    'no_data_value': numbaf32, 'extractor': numbastr, 'depth_tolerance': numbaf32,
                    'dist_exponent': numbaf64, 'inv_dist_exponent': numbaf64, 'dist_scale': numbaf64,
                    'var_scale': numbaf64, 'iho_fixed': numbaf64, 'iho_percent': numbaf64, 'median_length': numbai32,
                    'quotient_limit': numbaf64, 'max_hypothesis_ratio': numbaf32, 'discount': numbaf64,
                    'est_offset': numbaf64, 'bayes_factor_threshold': numbai64, 'runlength_threshold': numbai32,
                    'min_context': numbai32, 'max_context': numbai32, 'stddev_to_conf_scale': numbaf32,
                    'blunder_min': numbaf32, 'blunder_percent': numbaf32, 'blunder_scalar': numbaf32,
                    'capture_dist_scale': numbaf32, 'variance_selection': numbastr}


@jitclass(cube_params_spec)
class CubeParameters:
    """
    Construct the parameters object to feed the cube algorithm the different parameters
    """
    def __init__(self, iho_order: numbastr, grid_resolution_x: numbaf32, grid_resolution_y: numbaf32):
        iho_fixed, iho_percent = get_iho_limits(iho_order)
        dist_scale = min(grid_resolution_x, grid_resolution_y)
        min_context = min(1, int(5.0 / dist_scale))
        max_context = min(1, int(10.0 / dist_scale))
        dist_exponent = 2.0

        self.iho_order = iho_order
        self.grid_resolution_x = grid_resolution_x
        self.grid_resolution_y = grid_resolution_y

        self.no_data_value = np.float32(np.nan)  # Value used to indicate no data
        self.extractor = 'lhood'  # method used to extract information from sheet, one of 'lhood', 'prior', 'posterior', 'predsurf', 'union'
        self.depth_tolerance = 0.01  # the maximum difference allowed when searching hypotheses by depth
        self.dist_exponent = dist_exponent  # exponent on distance for variance scale
        self.inv_dist_exponent = 1 / self.dist_exponent  # inverse of dist exponent for efficiency
        self.dist_scale = dist_scale  # normalization coefficient for distance
        self.var_scale = dist_scale ** -dist_exponent  # variance scale dilution factor, placeholder, will be computed on initialization
        self.iho_fixed = iho_fixed  # fixed portion of IHO error budget, placeholder, will be computed on initialization
        self.iho_percent = iho_percent  # variable portion of IHO error budget, placeholder, will be computed on initialization
        self.median_length = 11  # Length of median pre-filter sort queue (must be odd number for algorithm)
        self.quotient_limit = 255.0  # Outlier quotient upper allowable limit, Approx. 0.1% F(1,6)
        self.max_hypothesis_ratio = 5.0  # ceiling to place on hypothesis strength ratios
        self.discount = 1.0  # Discount factor for evolution noise variance
        self.est_offset = 4.0  # Threshold for significant offset from current estimate to warrant an intervention, Set by West & Harrison's method of significant percentage points.
        self.bayes_factor_threshold = 0.135  # Bayes factor threshold for either a single estimate, or the worst case recent sequence to warrant an intervention, Set by West & Harrison's method of significant evidence for M_1
        self.runlength_threshold = 5  # Run length threshold for worst case recent sequence to indicate a drift failure and hence to warrant an intervention, Ball-park figure following West & Harrison's method
        self.min_context = min_context  # Minimum context search range for hypothesis disambiguation algorithm
        self.max_context = max_context  # Maximum context search range
        self.stddev_to_conf_scale = 1.96  # Scale from Std.Dev.to CI, 95 percent CI
        # blunders = beam solutions generated by the multibeam that do not correctly represent the seafloor
        self.blunder_min = 10.0  # Minimum depth difference from pred.depth to consider a blunder
        self.blunder_percent = 0.25  # Percentage of predicted depth to be considered a blunder, if more than the minimum (0 < p < 1, typ.0.25).
        self.blunder_scalar = 3.0  # Scale on initialisation surface std. dev. at a node to allow before considering deep spikes to be blunders.
        self.capture_dist_scale = 0.05  # Scale on predicted or estimated depth for how far out to accept data. (unitless; typically 0.05 for hydrography but can be greater for geological mapping in flat areas with sparse data)

        # Controls the reported variance, one of 'cube' to use CUBE's posterior variance estimate, 'input' to track and
        #   use input sample variance, and 'max' to report the greater of the two
        self.variance_selection = 'cube'


parameters_type = CubeParameters.class_type.instance_type


@njit
def return_default_cube_parameters(iho_order: str, grid_resolution_x: float, grid_resolution_y: float):
    cbp = CubeParameters(iho_order, grid_resolution_x, grid_resolution_y)
    return cbp


sounding_spec = {'depth': numbaf32, 'variance': numbaf32, 'vert_unc': numbaf32, 'horiz_unc': numbaf32}


@jitclass(sounding_spec)
class Sounding:
    def __init__(self, depth, variance, vert_unc, horiz_unc):
        self.depth = depth
        self.variance = variance
        self.vert_unc = vert_unc
        self.horiz_unc = horiz_unc


sounding_type = Sounding.class_type.instance_type


@njit
def return_new_sounding(depth: numbaf32, variance: numbaf32, vert_unc: numbaf32, horiz_unc: numbaf32):
    snd = Sounding(depth, variance, vert_unc, horiz_unc)
    return snd


hypothesis_spec = {'current_depth': numbaf32, 'current_variance': numbaf32, 'predict_depth': numbaf32,
                   'predict_variance': numbaf32, 'cum_bayes_fac': numbaf32, 'seq_length': numbai32,
                   'hypothesis_number': numbai32, 'number_of_samples': numbai32, 'variance_estimate': numbaf32}


@jitclass(hypothesis_spec)
class Hypothesis:
    def __init__(self, initial_mean_estimate, initial_variance_estimate):
        self.current_depth = initial_mean_estimate  # current depth mean estimate
        self.current_variance = initial_variance_estimate  # current depth variance estimate
        self.predict_depth = initial_mean_estimate  # current depth next-state mean prediction
        self.predict_variance = initial_variance_estimate  # current depth next-state variance prediction
        self.cum_bayes_fac = 1.0  # cumulative bayes factor for node monitoring
        self.seq_length = 0  # worst case sequence length for monitoring
        self.hypothesis_number = 0  # index term for debugging
        self.number_of_samples = 1  # number of points incorporated into this node
        self.variance_estimate = 0.0  # running estimate of variance of inputs


hypothesis_type = Hypothesis.class_type.instance_type


@njit
def return_new_hypothesis(initial_mean_estimate: np.float32, initial_variance_estimate: np.float32):
    hypo = Hypothesis(initial_mean_estimate, initial_variance_estimate)
    return hypo


hypolist_spec = {}
hypolist_type = deferred_type()
hypolist_spec['data'] = hypothesis_type
hypolist_spec['next_data'] = optional(hypolist_type)


@jitclass(hypolist_spec)
class HypothesisList:
    def __init__(self, data, next_data):
        self.data = data
        self.next_data = next_data

    def prepend(self, data):
        return HypothesisList(data, self)

    def append(self, data):
        cur = self
        while cur.next_data is not None:
            cur = cur.next_data
        cur.next_data = HypothesisList(data, None)

    def drop_first(self):
        return self.next_data

    def insert(self, data, index: int):
        added = False
        if index > 0:
            cur = self
            idx = 0
            while cur.next_data is not None:
                if idx == (index - 1):
                    new_node = HypothesisList(data, cur.next_data)
                    cur.next_data = new_node
                    added = True
                    break
                cur = cur.next_data
                idx += 1
            if not added:
                self.append(data)

    def remove(self, index: int):
        cur = self
        idx = 0
        while cur.next_data is not None:
            idx += 1
            if idx == index:
                nextcur = cur.next_data
                cur.next_data = nextcur.next_data
                return nextcur.data
            cur = cur.next_data
        return None

    def get_data(self):
        cur = self
        outdata = [self.data]
        while cur.next_data is not None:
            cur = cur.next_data
            outdata.append(cur.data)
        return outdata

    def get_item(self, index: int):
        cur = self
        if index == 0:
            return cur.data
        idx = 0
        while cur.next_data is not None:
            idx += 1
            cur = cur.next_data
            if idx == index:
                return cur.data
        return None

    def get_nearest_in_depth(self, nearest_depth: float, depth_tolerance: float):
        cur = self
        idx = 0
        cur_diff = abs(cur.data.current_depth - nearest_depth)
        if cur_diff <= depth_tolerance:
            cur_index = 0
        else:
            cur_index = -1
        while cur.next_data is not None:
            idx += 1
            cur = cur.next_data
            diff = abs(cur.data.current_depth - nearest_depth)
            if diff < cur_diff and diff <= depth_tolerance:
                cur_diff = diff
                cur_index = idx
        return cur_index

    def get_nearest_min_error(self, depth: float, variance: float):
        cur = self
        cur_index = 0
        idx = 0
        cur_min_error = abs((depth - cur.data.predict_depth) / np.sqrt(cur.data.predict_variance + variance))
        while cur.next_data is not None:
            idx += 1
            cur = cur.next_data
            min_error = abs((depth - cur.data.predict_depth) / np.sqrt(cur.data.predict_variance + variance))
            if min_error < cur_min_error:
                cur_min_error = min_error
                cur_index = idx
        return cur_index

    def get_max_sample(self):
        cur = self
        cur_index = 0
        idx = 0
        current_max_pointcount = cur.data.number_of_samples
        second_highest_count = 0
        while cur.next_data is not None:
            idx += 1
            cur = cur.next_data
            count = cur.data.number_of_samples
            if count > current_max_pointcount:
                cur_index = idx
                if current_max_pointcount > second_highest_count:
                    second_highest_count = current_max_pointcount
                current_max_pointcount = count
            elif count > second_highest_count:
                second_highest_count = count
        return cur_index, current_max_pointcount, second_highest_count


hypolist_type.define(HypothesisList.class_type.instance_type)


queuelist_spec = {}
queuelist_type = deferred_type()
queuelist_spec['data'] = sounding_type
queuelist_spec['next_data'] = optional(queuelist_type)


@jitclass(queuelist_spec)
class QueueList:
    def __init__(self, data, next_data):
        self.data = data
        self.next_data = next_data

    def prepend(self, data):
        return QueueList(data, self)

    def append(self, data):
        cur = self
        while cur.next_data is not None:
            cur = cur.next_data
        cur.next_data = QueueList(data, None)

    def drop_first(self):
        return self.next_data

    def insert(self, data, index: int):
        added = False
        if index > 0:
            cur = self
            idx = 0
            while cur.next_data is not None:
                if idx == (index - 1):
                    new_node = QueueList(data, cur.next_data)
                    cur.next_data = new_node
                    added = True
                    break
                cur = cur.next_data
                idx += 1
            if not added:
                self.append(data)

    def remove(self, index: int):
        cur = self
        idx = 0
        while cur.next_data is not None:
            idx += 1
            if idx == index:
                nextcur = cur.next_data
                cur.next_data = nextcur.next_data
                return nextcur.data
            cur = cur.next_data
        return None

    def get_data(self):
        cur = self
        outdata = [self.data]
        while cur.next_data is not None:
            cur = cur.next_data
            outdata.append(cur.data)
        return outdata

    def get_item(self, index: int):
        cur = self
        if index == 0:
            return cur.data
        idx = 0
        while cur.next_data is not None:
            idx += 1
            cur = cur.next_data
            if idx == index:
                return cur.data
        return None


queuelist_type.define(QueueList.class_type.instance_type)


node_spec = {'queue': optional(queuelist_type), 'n_queued': numbai64, 'hypotheses': optional(hypolist_type),
             'nominated': optional(typeof(return_new_hypothesis(0.0, 0.0))), 'predicted_depth': numbaf32,
             'predicted_variance': numbaf32, 'depth_tolerance': numbaf32, 'bayes_factor_threshold': numbaf32,
             'est_offset': numbaf32, 'runlength_threshold': numbai32, 'discount': numbaf32, 'quotient_limit': numbaf32,
             'max_hypothesis_ratio': numbaf32, 'median_length': numbai32, 'blunder_min': numbaf32, 'blunder_percent': numbaf32,
             'blunder_scalar': numbaf32, 'capture_dist_scale': numbaf32, 'var_scale': numbaf32, 'dist_exponent': numbaf32,
             'stddev_to_conf_scale': numbaf32, 'no_data_value': numbaf32, 'variance_selection': numbastr}


@jitclass(node_spec)
class CubeNode:
    def __init__(self):
        self.queue = None
        self.n_queued = 0
        self.hypotheses = None
        self.nominated = None
        self.predicted_depth = 0.0
        self.predicted_variance = 0.0

        self.depth_tolerance = 0.01
        self.bayes_factor_threshold = 0.135
        self.est_offset = 4.0
        self.runlength_threshold = 5
        self.discount = 1.0
        self.quotient_limit = 30.0
        self.max_hypothesis_ratio = 5.0
        self.median_length = 11
        self.blunder_min = 10.0
        self.blunder_percent = 0.25
        self.blunder_scalar = 3.0
        self.capture_dist_scale = 0.05
        self.var_scale = 0.0
        self.dist_exponent = 2.0
        self.stddev_to_conf_scale = 1.96
        self.no_data_value = np.float32(np.nan)
        self.variance_selection = 'cube'


node_type = CubeNode.class_type.instance_type


@njit
def return_new_cubenode():
    return CubeNode()


_test_grid = typed.List()
_test_grid_row = typed.List()
_test_grid_row.append(CubeNode())
_test_grid.append(_test_grid_row)
grid_attribute_type = typeof(_test_grid)

grid_spec = {'minimum_easting': numbaf64, 'maximum_northing': numbaf64, 'num_columns': numbai64,
             'num_rows': numbai64, 'resolution_x': numbaf64, 'resolution_y': numbaf64, 'params': parameters_type,
             'no_data_value': numbaf32, 'dist_scale': numbaf64, 'inv_dist_exponent': numbaf64, 'min_context': numbai32,
             'max_context': numbai32, 'iho_order': numbastr, 'iho_fixed': numbaf64, 'iho_percent': numbaf64,
             'grid': grid_attribute_type}


@jitclass(grid_spec)
class CubeGrid:
    """
    Grid contains the 2d list of CubeNodes which themselves contain the list of hypotheses.  Currently an issue with
    jitclass that this class can't be returned from an njit function, you get a pickle/serialization error, due to the
    nested jitclass.

    I've heard there is a way to accomplish this with structref in numba that would avoid this issue,
    I have not looked into this yet.
    """

    def __init__(self, minimum_easting: numbaf64, maximum_northing: numbaf64, num_columns: numbai64, num_rows: numbai64,
                 resolution_x: numbaf64, resolution_y: numbaf64, params: parameters_type):
        self.minimum_easting = minimum_easting
        self.maximum_northing = maximum_northing

        self.num_columns = num_columns
        self.num_rows = num_rows
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        self.params = params
        self.no_data_value = params.no_data_value
        self.dist_scale = params.dist_scale
        self.inv_dist_exponent = params.inv_dist_exponent
        self.min_context = params.min_context
        self.max_context = params.max_context
        self.iho_order = params.iho_order
        self.iho_fixed = params.iho_fixed
        self.iho_percent = params.iho_percent

        grid = typed.List()
        for row in range(self.num_rows):
            grid_row = typed.List()
            for column in range(self.num_columns):
                newnode = return_new_cubenode()
                newnode.bayes_factor_threshold = params.bayes_factor_threshold
                newnode.est_offset = params.est_offset
                newnode.runlength_threshold = params.runlength_threshold
                newnode.discount = params.discount
                newnode.quotient_limit = params.quotient_limit
                newnode.median_length = params.median_length
                newnode.blunder_min = params.blunder_min
                newnode.blunder_percent = params.blunder_percent
                newnode.blunder_scalar = params.blunder_scalar
                newnode.capture_dist_scale = params.capture_dist_scale
                newnode.var_scale = params.var_scale
                newnode.dist_exponent = params.dist_exponent
                newnode.stddev_to_conf_scale = params.stddev_to_conf_scale
                newnode.no_data_value = params.no_data_value
                newnode.variance_selection = params.variance_selection
                grid_row.append(newnode)
            grid.append(grid_row)
        self.grid = grid


grid_type = CubeNode.class_type.instance_type


@njit
def return_new_cubegrid(minimum_easting: numbaf64, maximum_northing: numbaf64, num_columns: numbai64, num_rows: numbai64,
                       resolution_x: numbaf64, resolution_y: numbaf64, params: parameters_type):
    # due to pickling errors in nested numba jitclasses, you cannot return this class and access it outside of njit functions
    # https://github.com/numba/numba/issues/6640
    cg = CubeGrid(minimum_easting, maximum_northing, num_columns, num_rows, resolution_x, resolution_y, params)
    return cg


@njit
def cube_node_new_hypothesis(node: CubeNode, new_sounding: Sounding):
    """
    Construct a new hypothesis, add to current list

    Parameters
    ----------
    node
        Node to add a hypothesis to
    new_sounding
        new sounding that we are adding to the node
    """

    new_hypo = return_new_hypothesis(new_sounding.depth, new_sounding.variance)
    if node.hypotheses is None:
        if Debug:
            print('cube_node_new_hypothesis: New hypothesis generated, no preexisting hypotheses')
        new_hypo.hypothesis_number = 1
        node.hypotheses = HypothesisList(new_hypo, None)
    else:
        new_hypo.hypothesis_number = len(node.hypotheses.get_data()) + 1
        node.hypotheses.append(new_hypo)
        if Debug:
            print('cube_node_new_hypothesis: New hypothesis generated, existing hypotheses found')


@njit
def cube_node_remove_hypothesis(node: CubeNode, depth: numbaf32):
    """
    This removes a hypothesis from a CubeNode permanently.  The hypothesis to remove is determined by the depth
    provided.  The algorithm allows up to node.depth_tolerance difference between this depth and the depth in the
    hypothesis, but will only remove the hypothesis if there is a unique match to the depth.  Tolerance is nominally
    a metric whisker (slightly smaller than the imperial), or 0.01m.

    Parameters
    ----------
    node
        Node to remove a hypothesis from
    depth
        the depth of the hypothesis to remove

    Returns
    -------
    Bool
        True if hypothesis was removed
    """

    if node.hypotheses is None:
        if Debug:
            print('cube_node_remove_hypothesis: unable to remove hypothesis, no hypotheses found')
        return False
    nearest_index = node.hypotheses.get_nearest_in_depth(depth, node.depth_tolerance)
    if nearest_index < 0:
        if Debug:
            print('cube_node_remove_hypothesis: unable to find hypothesis nearest in depth within depth tolerance')
        return False
    if nearest_index == 0 and node.hypotheses.next_data is None:
        hypo = node.hypotheses.data
        node.hypotheses = None
    elif nearest_index == 0:
        hypo = node.hypotheses.data
        node.hypotheses = node.hypotheses.drop_first()
    else:
        hypo = node.hypotheses.remove(nearest_index)
    if hypo is not None and node.nominated is not None and (hypo.number_of_samples == node.nominated.number_of_samples) and \
            (hypo.current_depth == node.nominated.current_depth):
        node.nominated = None
        if Debug:
            print('cube_node_remove_hypothesis: removed a nominated hypothesis, clearing nominated attribute')
    return True


@njit
def cube_node_nominate_hypothesis(node: CubeNode, depth: numbaf32):
    """
    This searches the list of hypotheses for one with depth within a whisker of the specified value --- in this
    case, a metric whisker, which is the same as 0.01m.  The hypothesis that matches, or the one that minimises
    the distance if there is more than one, is marked as 'nominated', and is reconstructed every time without
    running the disam. engine until the user explicitly resets the over-ride (with cube_node_reset_nomination) or
    more data is added to the node.

    Parameters
    ----------
    node
        Node to remove a hypothesis from
    depth
        the depth of the hypothesis to nominate

    Returns
    -------
    Bool
        True if hypothesis was nominated
    """

    if node.hypotheses is None:
        if Debug:
            print('cube_node_nominate_hypothesis: unable to nominate hypothesis, no hypotheses found')
        return False
    node.nominated = None
    nearest_index = node.hypotheses.get_nearest_in_depth(depth, node.depth_tolerance)
    if nearest_index < 0:
        if Debug:
            print('cube_node_nominate_hypothesis: unable to find hypothesis nearest in depth within depth tolerance')
        return False
    hypo = node.hypotheses.get_item(nearest_index)
    if hypo is None:
        if Debug:
            print('cube_node_nominate_hypothesis: unable to find hypothesis at nearest index')
        return False
    node.nominated = hypo
    if Debug:
        print('cube_node_nominate_hypothesis: successfully nominated existing hypothesis')
    return True


@njit
def cube_node_reset_nomination(node: CubeNode):
    """
    Remove the reference to the nominated hypothesis

    Returns
    -------
    bool
        True if nomination was cleared
    """

    if Debug:
        print('cube_node_reset_nomination: cleared nominated hypothesis')
    node.nominated = None
    return True


@njit
def cube_node_is_nominated(node: CubeNode):
    """
    Return True if there is a nominated hypothesis

    Returns
    -------
    bool
        if there is a nominated hypothesis, return True
    """

    if node.nominated is None:
        return False
    else:
        return True


@njit
def cube_node_set_preddepth(node: CubeNode, new_sounding: Sounding):
    """
    Set the node predicted depth.  Used to fix a node in place, hypothesis selection with 'predicted' will use this

    Parameters
    ----------
    node
        Node to set the predicted depth on
    new_sounding
        sounding data to use to set the predicted depth, variance
    """

    if Debug:
        print('cube_node_set_preddepth: set new predicted depth/variance')
    node.predicted_depth = new_sounding.depth
    node.predicted_variance = new_sounding.variance


@njit
def cube_node_monitor_hypothesis(node: CubeNode, hypo_index: numbai32, new_sounding: Sounding):
    """
    Compute West % Harrison's monitoring statistics for the node hypothesis.  Depends on node.est_offset (the offset
    we consider to be significant), node.bayes_factor_threshold (the Bayes factor threshold before intervention) and
    node.runlength_threshold (Number of bad factors to indicate sequence failure).

    Parameters
    ----------
    node
        Node containing the hypothesis we want to monitor
    hypo_index
        The index of the hypothesis we want to monitor
    new_sounding
        sounding data to use to compute the statistic

    Returns
    -------
    bool
        False if an intervention is required
    """

    if node.hypotheses is None:
        if Debug:
            print('cube_node_monitor_hypothesis: unable to monitor hypothesis, no hypotheses found')
        return False
    hypo = node.hypotheses.get_item(hypo_index)
    if hypo is None:
        if Debug:
            print('cube_node_monitor_hypothesis: unable to find hypothesis at index')
        return False
    forecast_variance = hypo.predict_variance + new_sounding.variance
    error = (new_sounding.depth - hypo.predict_depth) / np.sqrt(forecast_variance)

    if error >= 0:
        bayes_factor = np.exp(0.5 * (node.est_offset ** 2 - (2.0 * node.est_offset * error)))
    else:
        bayes_factor = np.exp(0.5 * (node.est_offset ** 2 + (2.0 * node.est_offset * error)))

    if bayes_factor < node.bayes_factor_threshold:
        if Debug:
            print('cube_node_monitor_hypothesis: bayes factor less than minimum threshold, outlier by single component Bayes factor')
        return False


    if hypo.cum_bayes_fac < 1.0:
        hypo.seq_length = hypo.seq_length + 1
    else:
        hypo.seq_length = 1
    hypo.cum_bayes_fac = bayes_factor * min(1.0, hypo.cum_bayes_fac)

    if (hypo.cum_bayes_fac < node.bayes_factor_threshold) or (hypo.seq_length > node.runlength_threshold):
        if Debug:
            print('cube_node_monitor_hypothesis: cum bayes fac < bayes_factor_threshold or seq length > runlength_threshold, potential outlier')
        return False
    if Debug:
        print('cube_node_monitor_hypothesis: no intervention required')
    return True


@njit
def cube_node_reset_monitor(node: CubeNode, hypo_index: numbai32):
    """
    Clear the monitoring data from the provided hypothesis

    Parameters
    ----------
    node
        Node containing the hypothesis we want to clear the monitoring data from
    hypo_index
        The index of the hypothesis we want to clear the monitor data from

    Returns
    -------
    bool
        True if the hypothesis was found and the monitoring data was cleared
    """
    if node.hypotheses is None:
        if Debug:
            print('cube_node_reset_monitor: unable to reset monitor, no hypotheses found')
        return False
    hypo = node.hypotheses.get_item(hypo_index)
    if hypo is None:
        if Debug:
            print('cube_node_reset_monitor: unable to find hypothesis at index')
        return False
    hypo.cum_bayes_fac = 1.0
    hypo.seq_length = 0
    if Debug:
        print('cube_node_reset_monitor: reset monitor')
    return True


@njit
def cube_node_update_hypothesis(node: CubeNode, hypo_index: numbai32, new_sounding: Sounding):
    """
    Update the given hypothesis (index is provided) being tracked at this node.  This implements the standard
    univariate dynamic linear model update equations (West & Harrison, 'Bayesian Forecasting and Dynamic Models',
    Springer, 2ed, 1997, Ch. 2), along with the Bayes factor monitoring code (W&H, Ch. 11).  The only failure mode
    possible with this code is if the input data would cause an intervention to be requested on the current track.
    In this case, it is the caller's responsibility to utilise the data point, since it will not be incorporated
    into the hypothesis --- typically this would mean adding a new hypothesis and pushing it onto the stack.

    Parameters
    ----------
    node
        Node containing the hypothesis we want to update
    hypo_index
        The index of the hypothesis we want to update
    new_sounding
        sounding data to use in updating

    Returns
    -------
    bool
        Returns False if the estimate does not really match the track that the hypothesis represents (i.e., an
        intervention is required).
    """

    if node.hypotheses is None:
        if Debug:
            print('cube_node_update_hypothesis: unable to update hypothesis, no hypotheses found')
        return False
    hypo = node.hypotheses.get_item(hypo_index)
    if hypo is None:
        if Debug:
            print('cube_node_update_hypothesis: unable to find hypothesis at index')
        return False
    monitoring_answer = cube_node_monitor_hypothesis(node, hypo_index, new_sounding)
    if not monitoring_answer:
        if Debug:
            print('cube_node_update_hypothesis: monitoring determined an intervention is required')
        return False
    if node.variance_selection != 'cube':
        hypo.variance_estimate = numbaf32((hypo.number_of_samples - 1) * hypo.variance_estimate / hypo.number_of_samples +
                                          (((new_sounding.depth - hypo.current_depth) ** 2) / hypo.number_of_samples))
    sys_variance = hypo.current_variance * numbaf32(1.0 - node.discount) / node.discount
    gain = hypo.predict_variance / numbaf32(new_sounding.variance + hypo.predict_variance)
    innovation = new_sounding.depth - hypo.predict_depth
    hypo.predict_depth = hypo.predict_depth + (gain * innovation)
    hypo.current_depth = hypo.predict_depth
    hypo.current_variance = new_sounding.variance * hypo.predict_variance / (new_sounding.variance + hypo.predict_variance)
    hypo.predict_variance = hypo.current_variance + sys_variance
    hypo.number_of_samples = hypo.number_of_samples + 1
    if Debug:
        print('cube_node_update_hypothesis: node hypothesis updated with new depth/variance')
    return True


@njit
def cube_node_best_hypothesis_index(node: CubeNode, new_sounding: Sounding):
    """
    Find the closest matching hypothesis in the current hypothesis list.  This computes the normalized absolute error
    between one-step forecast for each hypothesis currently being tracked and the input sample, and returns the index
    of the hypothesis with the smallest error value.

    If there is more than one node with the same error (unlikely in practice, but possible), then the last one in the
    list is chosen.

    Parameters
    ----------
    node
        Node containing the hypothesis we want to find the best hypothesis within
    new_sounding
        sounding data to use in querying

    Returns
    -------
    int
        index to the best hypothesis
    """
    if node.hypotheses is None:
        if Debug:
            print('cube_node_best_hypothesis_index: unable to query for best hypothesis index, no hypotheses found')
        return -1
    nearest_index = node.hypotheses.get_nearest_min_error(new_sounding.depth, new_sounding.variance)
    if Debug:
        print('cube_node_best_hypothesis_index: gathered new index for the best hypothesis')
    return nearest_index


@njit
def cube_node_update_node(node: CubeNode, new_sounding: Sounding):
    """
    Update the CUBE equations for this node and input.  This runs the basic filter equations, using the KF formulation,
    and its innovations formulation.  This algorithm now includes a discounted system noise variance model to set the
    evolution noise dynamically depending on the variance that was estimated at the previous stage (West & Harrison,
    'Bayesian Forecasting and Dynamic Models', Springer, 2ed., 1997, ch.2) and a monitoring scheme and feed-back
    interventions to allow the code to check that the estimates are staying in touch with the input data.  The
    monitoring scheme is also based on West & Harrison as above, Ch.11, Sec. 11.5.1, using cumulative Bayes factors
    and the unidirectional level shift alternate model.

    Parameters
    ----------
    node
        Node to update
    new_sounding
        sounding data to use in updating

    Returns
    -------
    bool
        True if node hypothesis was updated or if this is the first update.  False if a new hypothesis had to be
        created because data did not allow updating an existing one.
    """
    best_idx = cube_node_best_hypothesis_index(node, new_sounding)
    if best_idx == -1:
        # didn't match one, should only happen where there are no hypothesis, so we add a new one
        cube_node_new_hypothesis(node, new_sounding)
        if Debug:
            print('cube_node_update_node: no hypotheses found, so we added a new one')
    else:
        updated = cube_node_update_hypothesis(node, best_idx, new_sounding)
        if not updated:
            cube_node_reset_monitor(node, best_idx)
            cube_node_new_hypothesis(node, new_sounding)
            if Debug:
                print('cube_node_update_node: failed update, required intervention so we start a new hypothesis')
        else:
            if Debug:
                print('cube_node_update_node: existing best hypothesis successfully updated')
    return True


@njit
def cube_node_truncate(node: CubeNode):
    """
    Identify all points that are outliers and remove them from the queue.  The definition of 'outlier' depends on
    the self.quotient_limit attribute.  In general, the higher the value, the more extreme the difference between
    mean depth and point depth must be to be considered an outlier.

    In theory, the distribution of the quotient values computed should be approximately a Fisher F(1,N-2) where
    there are N points in the input sequence.  The values of the quotients are always positive, and monotonically
    increasing for worse outliers; therefore, one-sided critical values should be considered.

    Parameters
    ----------
    node
        Node to truncate
    """

    if node.n_queued < 3:
        print('cube_node_truncate: with too few queued points, truncate skipped')
        return
    if node.queue is None:
        if Debug:
            print('cube_node_truncate: unable to truncate, no queue found')
        return
    mean = 0.0
    sum_square_diff = 0.0
    num_points = node.n_queued - 1
    queue_data = node.queue.get_data()
    for nsounding in queue_data:
        mean += nsounding.depth
        sum_square_diff += nsounding.depth ** 2
    sum_square_diff -= (mean ** 2) / (num_points + 1)
    mean /= (num_points + 1)
    sum_square_diff_k = num_points * sum_square_diff / (num_points ** 2 - 1)

    outlier_index = np.zeros((num_points,), dtype=numbai32)
    outliers = 0
    for cnt, nsounding in enumerate(queue_data):
        diff_sq = (nsounding.depth - mean) ** 2
        quot = diff_sq / (sum_square_diff_k - (diff_sq / (num_points - 1)))
        if quot >= node.quotient_limit:
            if Debug:
                print('cube_node_truncate: point flagged for removal with quotient value greater than limit')
            outlier_index[outliers] = cnt
            outliers += 1

    # now we need to remove outliers from the last index to the first to not screw up the list order
    for i in outlier_index[:outliers][::-1]:
        node.queue.remove(i)
        node.n_queued = numbai64(node.n_queued - 1)


@njit
def cube_node_queue_flush_node(node: CubeNode):
    """
    This flushes the queue into the input sequence in order (i.e., take current median, resort, repeat).  Since the
    queue is always sorted, we can just walk the list in order, rather than having to re-sort or shift data, etc.
    When we have an even number of points, we take the shallowest of the points first; this means that we walk the
    list alternately to the left and right, starting to the right if the initial number of points is even, and to
    the left if the number of points is odd.  To avoid shifting the data, we just increase the step after every
    extraction, until we step off the LHS of the array.

    For a list of 10 points, the order ends up looking something like this: [4, 5, 3, 6, 2, 7, 1, 8, 0, 9]

    Parameters
    ----------
    node
        Node to flush
    """

    if node.n_queued == 0:
        print('cube_node_queue_flush_node: with no points, flush skipped')
        return
    scale = 1
    cube_node_truncate(node)
    if node.n_queued % 2 == 0:  # even
        ex_pt = int(node.n_queued / 2 - 1)
        direction = 1
    else:  # odd
        ex_pt = int(node.n_queued / 2)
        direction = -1
    while ex_pt >= 0:
        point = node.queue.get_item(ex_pt)
        cube_node_update_node(node, point)
        ex_pt += direction * scale
        direction = -direction
        scale += 1
    node.queue = None
    node.n_queued = numbai64(0)


@njit
def cube_node_choose_hypothesis(node: CubeNode):
    """
    Choose the best hypothesis for this node.  In this context, `best' means `hypothesis with most points',
    rather than through any other metric.  This may not be the `best' until all of the data is in, but it should
    give an idea of what's going on in the data structure at any point (particularly if it changes dramatically
    from sample to sample)

    Parameters
    ----------
    node
        Node to choose the hypothesis from

    Returns
    -------
    Hypothesis
        the hypothesis with the most points
    float
        the hypothesis strength ratio for this hypothesis (how convinced CUBE is that the hypo is good)
    """

    if node.hypotheses is None:
        if Debug:
            print('cube_node_choose_hypothesis: unable to query for best hypothesis index, no hypotheses found')
        return None, None
    hypo_ratio = 0.0
    idx, current_max_pointcount, second_highest_count = node.hypotheses.get_max_sample()
    if current_max_pointcount != 0 and second_highest_count != 0:
        hypo_ratio = max(0.0, node.max_hypothesis_ratio - (current_max_pointcount / second_highest_count))
    best_hypo = node.hypotheses.get_item(idx)
    return best_hypo, hypo_ratio


@njit
def cube_node_queue_fill(node: CubeNode, new_sounding: Sounding):
    """
    Insert a new point into the queue, maintain depth sorted order, with greater depths last

    Parameters
    ----------
    node
        Node to add the sounding to
    new_sounding
        new sounding data to add to the queue
    """
    if node.n_queued == 0:
        if Debug:
            print('cube_node_queue_fill: queue is empty, adding depth and variance to queue')
        node.queue = QueueList(new_sounding, None)
    else:
        insertion_index = None
        for i in range(node.n_queued):
            point = node.queue.get_item(i)
            if new_sounding.depth > point.depth:
                insertion_index = i + 1
            else:
                break

        if insertion_index is not None:
            if Debug:
                print('cube_node_queue_fill: inserted, adding depth, variance to queue')
            if insertion_index == 0:
                node.queue = node.queue.prepend(new_sounding)
            else:
                node.queue.insert(new_sounding, insertion_index)
        else:
            if Debug:
                print('cube_node_queue_fill: adding to beginning of queue, adding depth, variance to queue')
            node.queue = node.queue.prepend(new_sounding)
    node.n_queued = numbai64(node.n_queued + 1)


@njit
def cube_node_queue_insert(node: CubeNode, new_sounding: Sounding):
    """
    Insert a point in the already filled queue.  We then return the median point and insert this new point, ensuring
    that the queue remains sorted, with greater depths first.

    Parameters
    ----------
    node
        Node to add the sounding to
    new_sounding
        new sounding data to add to the queue

    Returns
    -------
    Sounding
        median sounding point data
    """

    median_index = int(np.floor(node.median_length / 2))
    m_sounding = node.queue.remove(median_index)
    insertion_index = None

    if new_sounding.depth > m_sounding.depth:
        check_range = range(median_index, node.n_queued - 1)
    else:
        check_range = range(node.n_queued - 1)

    for i in check_range:
        nextpoint = node.queue.get_item(i)
        if new_sounding.depth < nextpoint.depth:
            insertion_index = i
            break

    if insertion_index is not None:
        if Debug:
            print('cube_node_queue_insert: queue full, inserted at index, adding depth, variance to queue')
        if insertion_index == 0:
            node.queue = node.queue.prepend(new_sounding)
        else:
            node.queue.insert(new_sounding, insertion_index)
    else:
        if Debug:
            print('cube_node_queue_insert: queue full, adding to end of queue, adding depth, variance to queue')
        node.queue.append(new_sounding)

    # 2.56 being the 99 percent confidence interval for normal distribution
    first_pt = node.queue.get_item(0)
    last_pt = node.queue.get_item(node.n_queued - 1)
    low_water = last_pt.depth - 2.56 * np.sqrt(last_pt.variance)
    high_water = first_pt.depth + 2.56 * np.sqrt(first_pt.variance)

    if low_water >= high_water:  # confidence limits do not overlap
        cube_node_truncate(node)  # remove any outliers
    if Debug:
        print('cube_node_queue_insert: queue full, returning median point depth, variance')

    return m_sounding


@njit
def cube_node_add_to_queue(node: CubeNode, new_sounding: Sounding):
    """
    Insert points into the queue of estimates and insert point into the filter sequence if the queue is filled.

    This inserts the depth given into the queue associated with the	specified node, creating the queue if required.
    After the queue has been primed (i.e., filled with estimates), on each call this routine extracts the median
    value from the queue and then inserts it into the CUBE input sequence. Note that this algorithm means that the
    queue will always be full, and hence must be flushed before extracting any depth estimates (this can also be
    done to save memory).

    if use_queue is set to False, skips the queue entirely and runs update_node with the provided depth/variance

    Parameters
    ----------
    node
        Node to add the sounding to
    new_sounding
        new sounding data to add to the queue

    Returns
    -------
    bool
        True if data was added
    """

    if Debug:
        print('cube_node_add_to_queue: adding depth, variance to the queue')
    if node.queue is None:
        node.n_queued = numbai64(0)
    if node.n_queued < node.median_length:
        cube_node_queue_fill(node, new_sounding)
    else:
        median_sounding = cube_node_queue_insert(node, new_sounding)
        cube_node_update_node(node, median_sounding)
    return True


@njit
def cube_node_insert(node: CubeNode, new_sounding: Sounding, distance_to_node: numbaf32):
    """
    Insert a point into the node.  This will compute the variance scale factor for the new data, and send the data
    into the estimation queue.

    Parameters
    ----------
    node
        Node to add the sounding to
    new_sounding
        new sounding data added to the node
    distance_to_node
        distance from point to node

    Returns
    -------
    bool
        True if data was added
    """

    conf_95_percent = 1.96
    if np.isnan(node.predicted_depth):
        if Debug:
            print('cube_node_insert: Sounding rejected with predicted depth of NaN')
        return True
    dist = numbaf32(np.sqrt(distance_to_node))
    if node.predicted_depth:
        target_depth = node.predicted_depth
        # do the test for blunders here, since it makes no sense to test when there is no predicted depth
        # blunders = beam solutions generated by the multibeam that do not correctly represent the seafloor
        blunder_limit = min(target_depth - node.blunder_min, target_depth - node.blunder_percent * abs(target_depth))
        blunder_limit = min(blunder_limit, target_depth - node.blunder_scalar * np.sqrt(node.predicted_variance))
        if new_sounding.depth < blunder_limit:
            if Debug:
                print('cube_node_insert: Sounding rejected with predicted depth used, less than blunder limit')
            return True
    else:
        if Debug:
            print('cube_node_insert: Blunder limit test pass, no predicted depth for this node')
        target_depth = new_sounding.depth
    calculated_captdist = node.capture_dist_scale * abs(target_depth)
    if dist > max(calculated_captdist, 0.5):
        if Debug:
            print('cube_node_insert: sounding rejected, dist greater than max(0.5 or calculated capture distance)')
        return True
    if Debug:
        print('cube_node_insert: sounding accepted at node')
    # add horizontal positioning uncertainty, assumes 2sigma
    dist += conf_95_percent * np.sqrt(new_sounding.horiz_unc)
    # TODO this asked for range (range != 0) in the original source, don't have range
    sounding_range = 0.0
    if sounding_range != 0.0 and (not np.isnan(node.predicted_depth) and node.predicted_depth):
        offset = node.predicted_depth - new_sounding.depth
        if Debug:
            print('cube_node_insert: adding offset to depth')
    else:
        offset = 0.0

    # build new depth/variance
    new_sounding.depth += numbaf32(offset)
    new_sounding.variance = new_sounding.vert_unc * numbaf32(1.0 + node.var_scale * (dist ** node.dist_exponent))
    added = cube_node_add_to_queue(node, new_sounding)
    if not added:
        if Debug:
            print('cube_node_insert: failed to add to node')
        return False
    node.nominated = None
    return True


@njit
def cube_node_get_nominated_depth_uncertainty(node: CubeNode):
    """
    Return the nominated hypothesis answer

    Parameters
    ----------
    node
        node to draw the nominated hypothesis from

    Returns
    -------
    numbaf32
        depth value from the nominated hypothesis
    numbaf32
        uncertainty value from the nominated hypothesis
    numbaf32
        ratio value from the nominated hypothesis
    """

    if node.nominated is not None:
        depth = node.nominated.current_depth
        uncertainty = node.stddev_to_conf_scale * numbaf32(np.sqrt(node.nominated.current_variance))
        ratio = numbaf32(0.0)
        return depth, uncertainty, ratio
    else:
        if Debug:
            print('cube_node_get_nominated_depth_uncertainty: unable to get hypothesis answer, no samples found')
        return node.no_data_value, node.no_data_value, node.no_data_value


@njit
def cube_node_get_generic_hypothesis_depth_uncertainty(node: CubeNode, hypo: Hypothesis):
    """
    Return the hypothesis answer

    Parameters
    ----------
    node
        node to use in determining the cube parameters for this operation
    hypo
        hypothesis to draw the answer from

    Returns
    -------
    numbaf32
        depth value from the nominated hypothesis
    numbaf32
        uncertainty value from the nominated hypothesis
    numbaf32
        ratio value from the nominated hypothesis
    """

    if hypo.number_of_samples > 0:
        depth = hypo.current_depth
        if node.variance_selection == 'max':
            uncertainty = node.stddev_to_conf_scale * numbaf32(np.sqrt(max(hypo.current_variance, hypo.variance_estimate)))
        elif node.variance_selection == 'input':
            uncertainty = node.stddev_to_conf_scale * numbaf32(np.sqrt(hypo.variance_estimate))
        else:
            uncertainty = node.stddev_to_conf_scale * numbaf32(np.sqrt(hypo.current_variance))
        ratio = numbaf32(0.0)
        return depth, uncertainty, ratio
    else:
        if Debug:
            print('cube_node_get_generic_hypothesis_depth_uncertainty: unable to get hypothesis answer, no samples found')
        return node.no_data_value, node.no_data_value, node.no_data_value


@njit
def cube_node_extract_depth_uncertainty(node: CubeNode):
    """
    Return the hypothesis data for the current best estimate

    Parameters
    ----------
    node
        node containing the hypotheses

    Returns
    -------
    numbaf32
        depth value from the nominated hypothesis
    numbaf32
        uncertainty value from the nominated hypothesis
    numbaf32
        ratio value from the nominated hypothesis
    """

    if node.nominated is not None:
        if Debug:
            print('cube_node_extract_depth_uncertainty: getting the nominated hypothesis answer')
        depth, uncertainty, ratio = cube_node_get_nominated_depth_uncertainty(node)
        return depth, uncertainty, ratio
    elif node.hypotheses is None:
        if Debug:
            print('cube_node_extract_depth_uncertainty: no answer, no hypotheses found')
        return node.no_data_value, node.no_data_value, node.no_data_value
    elif node.hypotheses.next_data is None:  # only one hypothesis
        hypo = node.hypotheses.data
        depth, uncertainty, ratio = cube_node_get_generic_hypothesis_depth_uncertainty(node, hypo)
        if Debug:
            print('cube_node_extract_depth_uncertainty: getting the hypothesis answer, as only one was found')
        return depth, uncertainty, ratio
    else:
        best_hypo, hypo_ratio = cube_node_choose_hypothesis(node)
        if best_hypo is None:
            if Debug:
                print('cube_node_extract_depth_uncertainty: no answer, failed to gather best hypothesis')
            return node.no_data_value, node.no_data_value, node.no_data_value
        depth, uncertainty, ratio = cube_node_get_generic_hypothesis_depth_uncertainty(node, best_hypo)
        ratio = hypo_ratio  # use the ratio determined during hypothesis choosing
        if Debug:
            print('cube_node_extract_depth_uncertainty: getting the hypothesis answer, had to choose across multiple hypotheses')
        return depth, uncertainty, ratio


@njit
def cube_node_extract_closest_depth_uncertainty(node: CubeNode, query_depth: numbaf32, query_variance: numbaf32):
    """
    Return the hypothesis data for the closest hypothesis to the supplied query values in the minimum error sense

    Parameters
    ----------
    node
        node containing the hypotheses
    query_depth
        depth value to use for the query
    query_variance
        variance value to use for the query

    Returns
    -------
    numbaf32
        depth value from the nominated hypothesis
    numbaf32
        uncertainty value from the nominated hypothesis
    numbaf32
        ratio value from the nominated hypothesis
    """

    if (node.nominated is not None) or (node.hypotheses is None) or (node.hypotheses.next_data is None):
        if Debug:
            print('cube_node_extract_closest_depth_uncertainty: one or zero hypotheses found, defaulting to basic selection')
        depth, uncertainty, ratio = cube_node_extract_depth_uncertainty(node)
        return depth, uncertainty, ratio
    else:
        min_error = None
        nearest_hypo = None
        total_points = 0
        for hyp in node.hypotheses.get_data():
            if hyp.number_of_samples > 0:  # check that some data were used in making the hypothesis before accepting it as valid
                error = abs((hyp.current_depth - numbaf32(query_depth)) / numbaf32(np.sqrt(query_variance)))
                if min_error is None or error < min_error:
                    min_error = error
                    nearest_hypo = hyp
                total_points += hyp.number_of_samples
        if nearest_hypo is None:  # should never get to this point
            if Debug:
                print('cube_node_extract_closest_depth_uncertainty: no hypothesis found!')
            return node.no_data_value, node.no_data_value, node.no_data_value
        else:
            if Debug:
                print('cube_node_extract_closest_depth_uncertainty: chose nearest hypothesis')
            nearest_ratio = numbaf32(max(0.0, node.max_hypothesis_ratio - (nearest_hypo.number_of_samples / (total_points - nearest_hypo.number_of_samples))))
            depth, uncertainty, ratio = cube_node_get_generic_hypothesis_depth_uncertainty(node, nearest_hypo)
            ratio = nearest_ratio
            return depth, uncertainty, ratio


@njit
def cube_node_extract_posterior_depth_uncertainty(node: CubeNode, guide_depth: numbaf32, guide_variance: numbaf32):
    """
    Return the hypothesis data for the nearest hypothesis using a posterior weighted best depth estimate with the
    supplied guide depth and variance

    Parameters
    ----------
    node
        node containing the hypotheses
    guide_depth
        depth value to use for the query
    guide_variance
        variance value to use for the query

    Returns
    -------
    numbaf32
        depth value from the nominated hypothesis
    numbaf32
        uncertainty value from the nominated hypothesis
    numbaf32
        ratio value from the nominated hypothesis
    """

    if (node.nominated is not None) or (node.hypotheses is None) or (node.hypotheses.next_data is None):
        if Debug:
            print('cube_node_extract_posterior_depth_uncertainty: one or zero hypotheses found, defaulting to basic selection')
        depth, uncertainty, ratio = cube_node_extract_depth_uncertainty(node)
        return depth, uncertainty, ratio
    else:
        max_posterior = None
        nearest_hypo = None
        total_points = 0
        for hyp in node.hypotheses.get_data():
            if hyp.number_of_samples > 0:  # check that some data were used in making the hypothesis before accepting it as valid
                mean = hyp.current_depth
                posterior = -numbaf32((guide_depth - mean) ** 2) / numbaf32(2.0 * guide_variance) + numbaf32(np.log(hyp.number_of_samples))
                if max_posterior is None or posterior > max_posterior:
                    max_posterior = posterior
                    nearest_hypo = hyp
                total_points += hyp.number_of_samples
        if nearest_hypo is None:  # should never get to this point
            if Debug:
                print('cube_node_extract_posterior_depth_uncertainty: no hypothesis found!')
            return node.no_data_value, node.no_data_value, node.no_data_value
        else:
            if Debug:
                print('cube_node_extract_posterior_depth_uncertainty: chose nearest hypothesis')
            nearest_ratio = numbaf32(max(0.0, node.max_hypothesis_ratio - (nearest_hypo.number_of_samples / (total_points - nearest_hypo.number_of_samples))))
            depth, uncertainty, ratio = cube_node_get_generic_hypothesis_depth_uncertainty(node, nearest_hypo)
            ratio = nearest_ratio
            return depth, uncertainty, ratio


@njit
def cube_node_hypothesis_count(node: CubeNode):
    """
    Return the number of hypotheses in the node

    Parameters
    ----------
    node
        node containing the hypotheses

    Returns
    -------
    int
        number of hypotheses in this node
    """

    if node.hypotheses is None:
        return 0
    else:
        return len(node.hypotheses.get_data())


@njit
def cube_grid_insert_points(cg: CubeGrid, depth: np.ndarray, horizontal_uncertainty: np.ndarray, vertical_uncertainty: np.ndarray,
                            easting: np.ndarray, northing: np.ndarray):
    """
    Insert all provided points into the given CubeGrid

    Parameters
    ----------
    cg
        CubeGrid containing 2d list of CubeNodes that we want to add the points to
    depth
        1d array of depth values
    horizontal_uncertainty
        1d array of 2sigma horiz uncertainty values
    vertical_uncertainty
        1d array of 2sigma vert uncertainty values
    easting
        1d array of UTM easting values for the soundings
    northing
        1d array of UTM northing values for the soundings
    """

    conf_95_percent = 1.96
    conf_99_percent = 2.95
    for i in range(depth.shape[0]):
        max_variance_allowed = (cg.iho_fixed + cg.iho_percent * depth[i] ** 2) / (conf_95_percent ** 2)
        ratio = max_variance_allowed / vertical_uncertainty[i]
        if ratio <= 2.0:
            ratio = 2.0
        max_radius = conf_99_percent * np.sqrt(horizontal_uncertainty[i])
        radius = cg.dist_scale * (ratio - 1.0) ** cg.inv_dist_exponent - max_radius
        if radius < 0.0:
            radius = cg.dist_scale
        elif radius > max_radius:
            radius = max_radius
        if radius < cg.dist_scale:
            radius = cg.dist_scale
        # determine the coordinates of the effect square.  This is designed to compute the largest region the sounding
        # can effect, and hence to make the insertion more efficient by only offering the sounding where it is likely
        # to be used
        min_x = int(((easting[i] - radius) - cg.minimum_easting) / cg.resolution_x)
        max_x = int(((easting[i] + radius) - cg.minimum_easting) / cg.resolution_x)
        min_y = int((cg.maximum_northing - (northing[i] + radius)) / cg.resolution_y)
        max_y = int((cg.maximum_northing - (northing[i] - radius)) / cg.resolution_y)
        # check that the sounding hits somewhere in the grid
        if max_x < 0 or min_x >= (cg.num_columns - 1) or max_y < 0 or min_y >= (cg.num_rows - 1):
            if Debug:
                print('cube_grid_insert_points: Sounding out of bounds')
            continue  # out of bounds
        # clip to the interior of the current grid
        min_x = max(0, min_x)
        max_x = min(max_x, cg.num_columns - 1)
        min_y = max(0, min_y)
        max_y = min(max_y, cg.num_rows - 1)
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                node_x = cg.minimum_easting + x * cg.resolution_x
                node_y = cg.maximum_northing - y * cg.resolution_y
                distance_sq = (node_x - easting[i]) ** 2 + (node_y - northing[i]) ** 2
                if distance_sq >= radius ** 2:
                    if Debug:
                        print('cube_grid_insert_points: rejecting point as out of distance to node')
                    continue  # distance to great, not including this point in this node
                if Debug:
                    print('cube_grid_insert_points: adding point to node')
                snding = return_new_sounding(depth[i], 0.0, vertical_uncertainty[i], horizontal_uncertainty[i])
                cube_node_insert(cg.grid[y][x], snding, distance_sq)


@njit
def cube_grid_extract_data(cg: CubeGrid, method: numbastr):
    """
    Extract the hypothesis answer from all nodes in the grid, returning gridded products for the different
    values.

    Parameters
    ----------
    cg
        CubeGrid containing 2d list of CubeNodes that we want to extract the hypothesis answer from
    method
        method to use in determining the appropriate hypothesis value.  'local' to use the local spatial
        context to find the closest node with a single hypothesis and use that hypothesis depth to find the nearest
        hypothesis in terms of depth in the current node.  'prior' to use the hypothesis with the most points
        associated with it.  'posterior' to combine both prior and local methods to form an approximate Bayesian
        posterior distribution.  'predict' to get the hypothesis closest to the predicted depth associated with
        each node.

    Returns
    -------
    np.ndarray
        gridded depth values of shape (rows, columns) for the grid
    np.ndarray
        gridded uncertainty values of shape (rows, columns) for the grid
    np.ndarray
        gridded ratio values of shape (rows, columns) for the grid
    np.ndarray
        gridded hypothesis count values of shape (rows, columns) for the grid
    """

    depth = cg.no_data_value
    depth_grid = np.full((cg.num_rows, cg.num_columns), cg.no_data_value, dtype=np.float32)
    uncertainty = cg.no_data_value
    uncertainty_grid = np.full((cg.num_rows, cg.num_columns), cg.no_data_value, dtype=np.float32)
    ratio = cg.no_data_value
    ratio_grid = np.full((cg.num_rows, cg.num_columns), cg.no_data_value, dtype=np.float32)
    hypcnt = 0
    numhyp_grid = np.full((cg.num_rows, cg.num_columns), 0, dtype=np.int32)
    for row in range(cg.num_rows):
        for col in range(cg.num_columns):
            node = cg.grid[row][col]
            hypcnt = cube_node_hypothesis_count(node)
            if method == 'local' or method == 'posterior':
                if hypcnt <= 1:
                    depth, uncertainty, ratio = cube_node_extract_depth_uncertainty(node)
                else:
                    closest_node = None
                    for offset in range(cg.min_context, cg.max_context + 1):
                        target_rows = [row - offset, row + offset]
                        for target_row in target_rows:
                            if 0 <= target_row < cg.num_rows:
                                for col_offset in range(-offset, offset + 1):
                                    target_col = col + col_offset
                                    if target_col < 0 or target_col >= cg.num_columns:
                                        continue
                                    chk_node_hcount = cube_node_hypothesis_count(cg.grid[target_row][target_col])
                                    if chk_node_hcount == 1:
                                        closest_node = cg.grid[target_row][target_col]
                                        if Debug:
                                            print('cube_grid_extract_local: found closest node during row search')
                                        break
                            if closest_node is not None:
                                break
                        if closest_node is None:
                            target_cols = [col - offset, col + offset]
                            for target_col in target_cols:
                                if 0 <= target_col < cg.num_columns:
                                    for row_offset in range(-offset + 1, offset):
                                        target_row = row + row_offset
                                        if target_row < 0 or target_row >= cg.num_rows:
                                            continue
                                        chk_node_hcount = cube_node_hypothesis_count(cg.grid[target_row][target_col])
                                        if chk_node_hcount == 1:
                                            closest_node = cg.grid[target_row][target_col]
                                            if Debug:
                                                print('cube_grid_extract_local: found closest node during column search')
                                            break
                                if closest_node is not None:
                                    break
                        if closest_node is not None:
                            break
                    if closest_node is None:  # default to the basic node hypothesis selection, couldn't find a good hypothesis in the region
                        if Debug:
                            print("cube_grid_extract_local: default to the basic node hypothesis selection, couldn't find a good hypothesis in the region")
                        depth, uncertainty, ratio = cube_node_extract_depth_uncertainty(node)
                    else:
                        if Debug:
                            print("cube_grid_extract_local: extract value from closest node found")
                        depth, uncertainty, ratio = cube_node_extract_depth_uncertainty(closest_node)
                        if not np.isnan(depth) and not np.isnan(uncertainty):
                            if method == 'local':
                                depth, uncertainty, ratio = cube_node_extract_closest_depth_uncertainty(node, depth, uncertainty)
                            elif method == 'posterior':
                                depth, uncertainty, ratio = cube_node_extract_posterior_depth_uncertainty(node, depth, uncertainty)
            elif method == 'prior':
                depth, uncertainty, ratio = cube_node_extract_depth_uncertainty(node)
            elif method == 'predicted':
                depth, uncertainty, ratio = cube_node_extract_closest_depth_uncertainty(node, node.predicted_depth, node.predicted_variance)

            depth_grid[row][col] = depth
            uncertainty_grid[row][col] = uncertainty
            ratio_grid[row][col] = ratio
            numhyp_grid[row][col] = hypcnt
    return depth_grid, uncertainty_grid, ratio_grid, numhyp_grid


@njit(nogil=True)
def cube_grid_numba(depth: np.ndarray, horizontal_uncertainty: np.ndarray, vertical_uncertainty: np.ndarray,
                    easting: np.ndarray, northing: np.ndarray, num_columns: numbai64, num_rows: numbai64,
                    minimum_easting: numbaf64, maximum_northing: numbaf64, method: numbastr, params: CubeParameters):
    """
    Helper function for assembling the grid and running extract_data in njit space

    Parameters
    ----------
    depth
        1d array of depth values
    horizontal_uncertainty
        1d array of 2sigma horiz uncertainty values
    vertical_uncertainty
        1d array of 2sigma vert uncertainty values
    easting
        1d array of UTM easting values for the soundings
    northing
        1d array of UTM northing values for the soundings
    num_columns
        number of columns in the grid
    num_rows
        number of rows in the grid
    minimum_easting
        minimum easting value for the grid to determine origin
    maximum_northing
        maximum northing value for the grid to determine origin
    method
        method to use in determining the appropriate hypothesis value.  'local' to use the local spatial
        context to find the closest node with a single hypothesis and use that hypothesis depth to find the nearest
        hypothesis in terms of depth in the current node.  'prior' to use the hypothesis with the most points
        associated with it.  'posterior' to combine both prior and local methods to form an approximate Bayesian
        posterior distribution.  'predict' to get the hypothesis closest to the predicted depth associated with
        each node.
    params
        CubeParameters object to use in building the grid

    Returns
    -------
    np.ndarray
        gridded depth values of shape (rows, columns) for the grid
    np.ndarray
        gridded uncertainty values of shape (rows, columns) for the grid
    np.ndarray
        gridded ratio values of shape (rows, columns) for the grid
    np.ndarray
        gridded hypothesis count values of shape (rows, columns) for the grid
    """

    cg = return_new_cubegrid(minimum_easting, maximum_northing, num_columns, num_rows, params.grid_resolution_x,
                             params.grid_resolution_y, params)
    cube_grid_insert_points(cg, depth, horizontal_uncertainty, vertical_uncertainty, easting, northing)
    depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = cube_grid_extract_data(cg, method)
    return depth_grid, uncertainty_grid, ratio_grid, numhyp_grid


def run_cube_gridding(depth: np.ndarray, horizontal_uncertainty: np.ndarray, vertical_uncertainty: np.ndarray,
                      easting: np.ndarray, northing: np.ndarray, num_columns: int, num_rows: int, minimum_easting: float,
                      maximum_northing: float, method: str, iho_order: str, grid_resolution_x: float, grid_resolution_y: float,
                      **kwargs):
    """
    Entrance point in numba_cube, run this to run Cube.

    Grid contains the 2d list of CubeNodes which themselves contain the list of hypotheses.  Currently an issue with
    jitclass that this class can't be returned from an njit function, you get a pickle/serialization error, due to the
    nested jitclass.

    I've heard there is a way to accomplish this with structref in numba that would avoid this issue,
    I have not looked into this yet.

    Parameters
    ----------
    depth
        1d array of depth values
    horizontal_uncertainty
        1d array of 2sigma horiz uncertainty values
    vertical_uncertainty
        1d array of 2sigma vert uncertainty values
    easting
        1d array of UTM easting values for the soundings
    northing
        1d array of UTM northing values for the soundings
    num_columns
        number of columns in the grid
    num_rows
        number of rows in the grid
    minimum_easting
        minimum easting value for the grid to determine origin
    maximum_northing
        maximum northing value for the grid to determine origin
    method
        method to use in determining the appropriate hypothesis value.  'local' to use the local spatial
        context to find the closest node with a single hypothesis and use that hypothesis depth to find the nearest
        hypothesis in terms of depth in the current node.  'prior' to use the hypothesis with the most points
        associated with it.  'posterior' to combine both prior and local methods to form an approximate Bayesian
        posterior distribution.  'predict' to get the hypothesis closest to the predicted depth associated with
        each node.
    iho_order
        string representation of one of the IHO order categories, i.e. 'special' or 'order1a'
    grid_resolution_x
        grid resolution in easting (column) direction in meters
    grid_resolution_y
        grid resolution in northing (row) direction in meters
    kwargs
        keyword arguments used to modify cube parameters

    Returns
    -------
    np.ndarray
        gridded depth values of shape (rows, columns) for the grid
    np.ndarray
        gridded uncertainty values of shape (rows, columns) for the grid
    np.ndarray
        gridded ratio values of shape (rows, columns) for the grid
    np.ndarray
        gridded hypothesis count values of shape (rows, columns) for the grid
    """

    cp = return_default_cube_parameters(iho_order, grid_resolution_x, grid_resolution_y)
    for kpam, kval in kwargs.items():
        if kpam in cp.__dir__():
            setattr(cp, kpam, kval)
    if method in ['local', 'posterior', 'prior', 'predicted']:
        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = cube_grid_numba(depth, horizontal_uncertainty, vertical_uncertainty,
                                                                                easting, northing, num_columns, num_rows,
                                                                                minimum_easting, maximum_northing, method, cp)
    else:
        raise NotImplementedError(f"run_cube_gridding: {method} not supported, expected one of 'local', 'posterior', 'prior', 'predicted'")
    return depth_grid, uncertainty_grid, ratio_grid, numhyp_grid


def compile_now():
    """
    Numba employs justintime compiling that will make the first run of the code in this file pretty slow.  You can avoid
    a length first run by running this function to compile ahead of time.

    There is also an option to compile ahead of time and distribute pyd/so files, I haven't looked into it much yet.
    """
    numpoints = 10
    x = np.random.uniform(low=403744.0, high=403776.0, size=numpoints)
    y = np.random.uniform(low=4122665.0, high=4122688.0, size=numpoints)
    z = np.random.uniform(low=13.0, high=15.0, size=numpoints)
    tvu = np.random.uniform(low=0.1, high=1.0, size=numpoints)
    thu = np.random.uniform(low=0.3, high=1.3, size=numpoints)

    run_cube_gridding(z, thu, tvu, x, y, 32, 32, min(x), max(y), 'local', 'order1a', 1.0, 1.0)


if __name__ == '__main__':
    from time import perf_counter

    compile_now()

    starttime = perf_counter()
    print('****Start****')
    _numpoints = 100000
    _x = np.random.uniform(low=403744.0, high=403776.0, size=_numpoints)
    _y = np.random.uniform(low=4122665.0, high=4122688.0, size=_numpoints)
    _z = np.random.uniform(low=13.0, high=15.0, size=_numpoints)
    _tvu = np.random.uniform(low=0.1, high=1.0, size=_numpoints)
    _thu = np.random.uniform(low=0.3, high=1.3, size=_numpoints)
    _numrows, _numcols = (32, 32)
    _resolution_x, _resolution_y = (1.0, 1.0)
    _depth_grid, _uncertainty_grid, _ratio_grid, _numhyp_grid = run_cube_gridding(_z, _thu, _tvu, _x, _y, _numcols, _numrows,
                                                                                  min(_x), max(_y), 'local', 'order1a',
                                                                                  _resolution_x, _resolution_y)
    endtime = perf_counter()
    print('****CUBE complete: {}****'.format((endtime - starttime)))
