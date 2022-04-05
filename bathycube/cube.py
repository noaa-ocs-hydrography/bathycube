"""
Python implementation of the CUBE module, Combined Uncertainty and Bathymetry Estimator.  Python implementation done
by Eric Younkin, Feb 2022.

CUBE was developed as a research project within the Center of for Coastal and Ocean Mapping and NOAA/UNH Joint Hydrographic
Center (CCOM/JHC) at the University of New Hampshire, starting in the fall of 2000.
"""

import sys
import numpy as np
from numba import types, typed
from numba.experimental import jitclass
import json
from enum import Enum
import logging

# ex using jitclass with composition: https://stackoverflow.com/questions/38682260/how-to-nest-numba-jitclass


class StdErrFilter(logging.Filter):
    """
    filter out messages that are not CRITICAL or ERROR or WARNING
    """
    def filter(self, rec):
        return rec.levelno in (logging.CRITICAL, logging.ERROR, logging.WARNING)


class StdOutFilter(logging.Filter):
    """
    filter out messages that are not DEBUG or INFO
    """
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)


def return_logger(logfile: str = None, loglevel=logging.INFO):
    """
    If logfile is included, the file handler is added to the log so that the output is also driven to file.

    I disable the root logger by clearing out it's handlers because it always gets a default stderr log handler that
    ends up duplicating messages.  Since I want the stderr messages formatted nicely, I want to setup that handler \
    myself.

    Parameters
    ----------
    logfile
        path to the log file where you want the output driven to, if None, will not log to file
    loglevel
        logging level to use

    Returns
    -------
    logger: logging.Logger instance for the provided name/logfile

    """
    
    fmat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger('cube')
    logger.setLevel(loglevel)

    consolelogger = logging.StreamHandler(sys.stdout)
    consolelogger.setLevel(loglevel)
    # consolelogger.setFormatter(logging.Formatter(fmat))
    consolelogger.addFilter(StdOutFilter())

    errorlogger = logging.StreamHandler(sys.stderr)
    errorlogger.setLevel(logging.WARNING)
    # errorlogger.setFormatter(logging.Formatter(fmat))
    errorlogger.addFilter(StdErrFilter())

    logger.addHandler(consolelogger)
    logger.addHandler(errorlogger)

    if logfile is not None:
        filelogger = logging.FileHandler(logfile)
        filelogger.setLevel(loglevel)
        filelogger.setFormatter(logging.Formatter(fmat))
        logger.addHandler(filelogger)

    # eliminate the root logger handlers, it will have a default stderr pointing handler that ends up duplicating all the logs to console
    logging.getLogger().handlers = []

    return logger


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


class CubeParameters:
    def __init__(self):
        self.iho_order = 'order1a'
        self.grid_resolution_x = 0.0
        self.grid_resolution_y = 0.0

        self.initialization_interlock = None  # system mapsheet initialization marker
        self.no_data_value = np.float32(np.nan)  # Value used to indicate no data
        self.extractor = 'lhood'  # method used to extract information from sheet, one of 'lhood', 'prior', 'posterior', 'predsurf', 'union'
        self.nodata_depth = 0.0  # depth to initialize estimates
        self.nodata_variance = 1000000  # variance value for initialization
        self.dist_exponent = 2.0  # exponent on distance for variance scale
        self.inv_dist_exponent = 1 / self.dist_exponent  # inverse of dist exponent for efficiency
        self.dist_scale = 0.0  # normalization coefficient for distance
        self.var_scale = 0.0  # variance scale dilution factor, placeholder, will be computed on initialization
        self.iho_fixed = 0.0  # fixed portion of IHO error budget, placeholder, will be computed on initialization
        self.iho_percent = 0.0  # variable portion of IHO error budget, placeholder, will be computed on initialization
        self.median_length = 11  # Length of median pre-filter sort queue (must be odd number for algorithm)
        self.quotient_limit = 30.0  # Outlier quotient upper allowable limit, Approx. 0.1% F(1,6)
        self.discount = 1.0  # Discount factor for evolution noise variance
        self.est_offset = 4.0  # Threshold for significant offset from current estimate to warrant an intervention, Set by West & Harrison's method of significant percentage points.
        self.bayes_factor_threshold = 0.135  # Bayes factor threshold for either a single estimate, or the worst case recent sequence to warrant an intervention, Set by West & Harrison's method of significant evidence for M_1
        self.runlength_threshold = 5  # Run length threshold for worst case recent sequence to indicate a drift failure and hence to warrant an intervention, Ball-park figure following West & Harrison's method
        self.min_context = 5  # Minimum context search range for hypothesis disambiguation algorithm
        self.max_context = 10  # Maximum context search range
        self.stddev_to_conf_scale = 1.96  # Scale from Std.Dev.to CI, 95 percent CI
        # blunders = beam solutions generated by the multibeam that do not correctly represent the seafloor
        self.blunder_min = 10.0  # Minimum depth difference from pred.depth to consider a blunder
        self.blunder_percent = 0.25  # Percentage of predicted depth to be considered a blunder, if more than the minimum (0 < p < 1, typ.0.25).
        self.blunder_scalar = 3.0  # Scale on initialisation surface std. dev. at a node to allow before considering deep spikes to be blunders.
        self.capture_dist_scale = 0.05  # Scale on predicted or estimated depth for how far out to accept data. (unitless; typically 0.05 for hydrography but can be greater for geological mapping in flat areas with sparse data)

        # Controls the reported variance, one of 'cube' to use CUBE's posterior variance estimate, 'input' to track and
        #   use input sample variance, and 'max' to report the greater of the two
        self.variance_selection = 'cube'

    def _get_iho_limits(self):
        """
        Get fixed and variable Total Vertical Uncertainty components for the different IHO Order categories, see S-44
        Table 1 - Minimum Bathymetry Standards for Safety of Navigation Hydrographic Surveys

        Returns
        -------
        float
            'a' component, the fixed component of the TVU equation
        float
            'b' component, the variable component of the TVU equation
        """
        return get_iho_limits(self.iho_order)

    def initialize(self, iho_order: str = 'order1a', grid_resolution_x: float = 1.0, grid_resolution_y: float = 1.0):
        """
        Build the situational parameters now, those related to IHO order or grid resolution.

        Parameters
        ----------
        iho_order
            one of the IHO order string identifiers, i.e. 'order1a'
        grid_resolution_x
            size of the grid cell in the x/easting direction
        grid_resolution_y
            size of the grid cell in the y/northing direction

        Returns
        -------

        """
        self.iho_order = iho_order
        self.grid_resolution_x = grid_resolution_x
        self.grid_resolution_y = grid_resolution_y
        # Compute distance scale based on node spacing
        self.dist_scale = min(grid_resolution_x, grid_resolution_y)  # normalization coefficient for distance
        self.min_context = min(1, int(self.min_context / self.dist_scale))
        self.max_context = min(1, int(self.max_context / self.dist_scale))
        # Compute variance scaling factor for dilution function
        self.var_scale = self.dist_scale ** -self.dist_exponent
        # IHO Survey Order limits for determining maximum allowable error
        self.iho_fixed, self.iho_percent = self._get_iho_limits()

        # we square these?  Not sure about this but it is in the C code
        self.iho_fixed = self.iho_fixed ** 2
        self.iho_percent = self.iho_percent ** 2

    def write_parameter_file(self, param_file: str):
        try:
            with open(param_file, 'w') as outfile:
                json.dump(self.__dict__, outfile)
                print('New CubeParameters file written to {}'.format(param_file))
        except:
            raise ValueError('CubeParameters: Unable to write new parameter file to {}'.format(param_file))

    def open_parameter_file(self, param_file: str):
        valid_data = False
        with open(param_file, 'r') as infile:
            try:
                data = json.load(infile)
            except:
                raise ValueError('CubeParameters: Unable to read data from {} as json'.format(param_file))
            for ky, val in data.items():
                if ky in self.__dict__:
                    self.__setattr__(ky, val)
                    valid_data = True
            if valid_data:
                print('CubeParameters read successfully from {}'.format(param_file))
            else:
                print('CubeParameters: Unable to find any valid data in {}'.format(param_file))


class Hypothesis:
    def __init__(self, initial_mean_estimate=0.0, initial_variance_estimate=0.0):
        self.current_depth = initial_mean_estimate  # current depth mean estimate
        self.current_variance = initial_variance_estimate  # current depth variance estimate
        self.predict_depth = initial_mean_estimate  # current depth next-state mean prediction
        self.predict_variance = initial_variance_estimate  # current depth next-state variance prediction
        self.cum_bayes_fac = 1.0  # cumulative bayes factor for node monitoring
        self.seq_length = 0  # worst case sequence length for monitoring
        self.hypothesis_number = 0  # index term for debugging
        self.number_of_points = 1  # number of points incorporated into this node
        self.variance_estimate = 0.0  # running estimate of variance of inputs


class CubeNode:
    """
    CubeNode - The primary estimation structural element.  This maintains the median pre-
    filter queue, the linked list of depth hypotheses, and the sample statistics
    for a single node.
    """

    def __init__(self, depth_tolerance: float = 0.01, bayes_factor_threshold: float = 0.135, est_offset: float = 4.0,
                 runlength_threshold: int = 5, discount: float = 1.0, quotient_limit: float = 30.0,
                 max_hypothesis_ratio: float = 5.0, median_length: float = 11, blunder_min: float = 10.0,
                 blunder_percent: float = 0.25, blunder_scalar: float = 3.0, capture_dist_scale: float = 0.05,
                 var_scale: float = 0.0, dist_exponent: float = 2.0, stddev_to_conf_scale: float = 1.96,
                 no_data_value: float = np.float32(np.nan), variance_selection: str = 'cube', use_queue: bool = True,
                 logger: logging.Logger = logging.getLogger()):
        """
        These default arguments are used or otherwise driven from the CubeParams class

        Parameters
        ----------
        depth_tolerance
            the maximum difference allowed when searching hypotheses by depth
        bayes_factor_threshold
            Bayes factor threshold for either a single estimate, or the worst case recent sequence to warrant an
            intervention, Set by West & Harrison's method of significant evidence for M_1
        est_offset
            Threshold for significant offset from current estimate to warrant an intervention, Set by West & Harrison's
            method of significant percentage points.
        runlength_threshold
            Run length threshold for worst case recent sequence to indicate a drift failure and hence to warrant an
            intervention, Ball-park figure following West & Harrison's method
        discount
            Discount factor for evolution noise variance
        quotient_limit
            Outlier quotient upper allowable limit, Approx. 0.1% F(1,6).  From CUBE User Manual - With the released
            multiple-hypothesis version of CUBE this parameter is no longer necessary, and could, if set inappropriately
            low, eliminate valid soundings from consideration. It is now a dangerous parameter, rather than a useful one.
            Hence it should always either be set to its maximum value of 255, or else removed as a user-accessible parameter.
        max_hypothesis_ratio
            ceiling to place on hypothesis strength ratios
        median_length
            Length of median pre-filter sort queue (must be odd number for algorithm)
        blunder_min
            Minimum depth difference from pred.depth to consider a blunder
        blunder_percent
            Percentage of predicted depth to be considered a blunder, if more than the minimum (0 < p < 1, typ.0.25)
        blunder_scalar
            Scale on initialisation surface std. dev. at a node to allow before considering deep spikes to be blunders
        capture_dist_scale
            Scale on predicted or estimated depth for how far out to accept data. (unitless; typically 0.05 for hydrography
            but can be greater for geological mapping in flat areas with sparse data)
        var_scale
            variance scale dilution factor
        dist_exponent
            exponent on distance for variance scale
        stddev_to_conf_scale
            Scale from Std.Dev.to CI, 95 percent CI is the default
        no_data_value
            Value used to indicate no data
        variance_selection
            controls the reported variance, one of 'cube' to use CUBE's posterior variance estimate, 'input' to track and
            use input sample variance, and 'max' to report the greater of the two
        use_queue
            Executes the 'Reordering' step, see CUBE User Manual 3.1.  With this set to False, this step is skipped.
            User Manual states that with multiple hypothesis implementation of CUBE, Reordering is no longer necessary.
        logger
            logger instance for debug messaging
        """

        self.queue = []
        self.n_queued = 0
        self.hypotheses = []  # this should be a list of Hypothesis
        self.nominated = None
        self._pred_depth = 0.0
        self._pred_var = 0.0

        self.depth_tolerance = depth_tolerance
        self.bayes_factor_threshold = bayes_factor_threshold
        self.est_offset = est_offset
        self.runlength_threshold = runlength_threshold
        self.discount = discount
        self.quotient_limit = quotient_limit
        self.max_hypothesis_ratio = max_hypothesis_ratio
        self.median_length = median_length
        self.blunder_min = blunder_min
        self.blunder_percent = blunder_percent
        self.blunder_scalar = blunder_scalar
        self.capture_dist_scale = capture_dist_scale
        self.var_scale = var_scale
        self.dist_exponent = dist_exponent
        self.stddev_to_conf_scale = stddev_to_conf_scale
        self.no_data_value = no_data_value
        self.variance_selection = variance_selection.lower()
        self.use_queue = use_queue

        self.logger = logger

    @property
    def predicted_depth(self):
        return self._pred_depth

    @predicted_depth.setter
    def predicted_depth(self, new_depth: float):
        self._pred_depth = new_depth

    @property
    def predicted_variance(self):
        return self._pred_var

    @predicted_variance.setter
    def predicted_variance(self, new_variance: float):
        self._pred_var = new_variance

    def add_hypothesis(self, depth: float, variance: float, null_hypothesis: bool = False):
        """
        Add a specific depth hypothesis to the current list

        Parameters
        ----------
        depth
            depth to set for the hypothesis
        variance
            variance to set for the hypothesis
        null_hypothesis
            if True, this is a null hypothesis, which is a specific hypothesis that has the number of points set to zero
        """

        new_hypo = Hypothesis(depth, variance)
        if null_hypothesis:
            new_hypo.number_of_points = 0
        new_hypo.hypothesis_number = len(self.hypotheses) + 1
        self.logger.log(logging.DEBUG, f'add_hypothesis: new hypothesis number {new_hypo.hypothesis_number} for depth {depth} variance {variance}')
        self.hypotheses.append(new_hypo)

    def remove_hypothesis(self, depth: float):
        """
        This removes a hypothesis from a CubeNode permanently.  The hypothesis to remove is determined by the depth
        provided.  The algorithm allows up to self.depth_tolerance difference between this depth and the depth in the
        hypothesis, but will only remove the hypothesis if there is a unique match to the depth.  Tolerance is nominally
        a metric whisker (slightly smaller than the imperial), or 0.01m.

        Parameters
        ----------
        depth
            the depth of the hypothesis to remove
        """

        hypo_idx = [self.hypotheses.index(h) for h in self.hypotheses if (abs(depth - h.current_depth) < self.depth_tolerance)]
        if len(hypo_idx) == 0:
            self.logger.log(logging.WARNING, 'remove_hypothesis: unable to remove hypothesis at depth {}, no hypothesis found within {} meters'.format(depth, self.depth_tolerance))
        elif len(hypo_idx) == 1:
            hypo_idx = hypo_idx[0]
            if self.nominated is not None and (self.nominated == self.hypotheses[hypo_idx]):
                self.nominated = None
            self.hypotheses.pop(hypo_idx)
            self.logger.log(logging.DEBUG, f'remove_hypothesis: hypothesis number {hypo_idx} removed')
        else:
            self.logger.log(logging.ERROR, 'remove_hypothesis: Found multiple hypothesis at depth {} +- {}, unable to remove a single hypothesis'.format(depth, self.depth_tolerance))
            raise ValueError('remove_hypothesis: Found multiple hypothesis at depth {} +- {}, unable to remove a single hypothesis'.format(depth, self.depth_tolerance))

    def nominate_hypothesis(self, depth: float):
        """
        This searches the list of hypotheses for one with depth within a whisker of the specified value --- in this
        case, a metric whisker, which is the same as 0.01m.  The hypothesis that matches, or the one that minimises
        the distance if there is more than one, is marked as 'nominated', and is reconstructed every time without
        running the disam. engine until the user explicitly resets the over-ride (with cube_node_reset_nomination) or
        more data is added to the node.

        Parameters
        ----------
        depth
            depth of the hypothesis that we want to preserve by nominating
        """

        min_depth_distance = None
        curr_hypo = None
        for hypo in self.hypotheses:
            depth_difference = abs(depth - hypo.current_depth)
            if depth_difference < self.depth_tolerance:
                if min_depth_distance:  # this is not the first hypothesis that we have found within the tolerance
                    if depth_difference < min_depth_distance:
                        min_depth_distance = depth_difference
                        curr_hypo = hypo
                        self.logger.log(logging.DEBUG, f'nominate_hypothesis: clearing previously selected hypothesis for hypothesis, selecting hypothesis at depth {depth}')
                else:  # this is the first hypo found within the tolerance
                    min_depth_distance = depth_difference
                    curr_hypo = hypo
                    self.logger.log(logging.DEBUG, f'nominate_hypothesis: selecting hypothesis at depth {depth}')
        self.nominated = curr_hypo
        if self.nominated is None:
            self.logger.log(logging.WARNING, 'nominate_hypothesis: Warning, no hypothesis found to nominate at depth {} +- {}'.format(depth, self.depth_tolerance))

    def clear_nomination(self):
        """
        Remove the reference to the nominated hypothesis
        """

        self.nominated = None
        self.logger.log(logging.DEBUG, 'clear_nomination: remove nominated hypothesis')

    def has_nomination(self):
        """
        Return True if there is a nominated hypothesis

        Returns
        -------
        bool
            if there is a nominated hypothesis, return True
        """

        if self.nominated is not None:
            return True
        else:
            return False

    def monitor_hypothesis(self, hypo_index: int, new_depth: float, new_variance: float):
        """
        Compute West % Harrison's monitoring statistics for the node hypothesis.  Depends on self.est_offset (the offset
        we consider to be significant), self.bayes_factor_threshold (the Bayes factor threshold before intervention) and
        self.runlength_threshold (Number of bad factors to indicate sequence failure).

        Parameters
        ----------
        hypo_index
            The index of the hypothesis we want to monitor
        new_depth
            new input sample which is about to be incorporated
        new_variance
            observation noise variance

        Returns
        -------
        bool
            False if an intervention is required
        """

        try:
            hypo = self.hypotheses[hypo_index]
        except IndexError:
            self.logger.log(logging.ERROR, f'monitor_hypothesis: Unable to pull hypothesis at index {hypo_index}')
            return False

        forecast_variance = hypo.predict_variance + new_variance
        error = (new_depth - hypo.predict_depth) / np.sqrt(forecast_variance)

        # the est_offset is W&H's `h' parameter (i.e., expected normalised difference between the current forecast and
        #   the observation which just indicates an outlier)
        if error >= 0:
            bayes_factor = np.exp(0.5 * (self.est_offset ** 2 - (2.0 * self.est_offset * error)))
        else:
            bayes_factor = np.exp(0.5 * (self.est_offset ** 2 + (2.0 * self.est_offset * error)))
        self.logger.log(logging.DEBUG, f'monitor_hypothesis: calculated bayes factor {bayes_factor}, error {error}, forecast variance {forecast_variance}')

        # check for single component failure
        # The bayes_factor_threshold is W&H's `tau' (i.e., the minimum Bayes factor which is acceptable as evidence for the current model)
        if bayes_factor < self.bayes_factor_threshold:
            self.logger.log(logging.DEBUG, f'monitor_hypothesis: bayes factor less than minimum threshold {self.bayes_factor_threshold}, potential outlier')
            return False
        # update monitors
        if hypo.cum_bayes_fac < 1.0:
            hypo.seq_length += 1
        else:
            hypo.seq_length = 1
        hypo.cum_bayes_fac = bayes_factor * min(1.0, hypo.cum_bayes_fac)
        # check for consecutive failure errors
        # The runlength_t is W&H's limit on l_t (i.e., the number of consequtively bad Bayes factors which indicate that there has been a gradual shift away from the predictor)
        if (hypo.cum_bayes_fac < self.bayes_factor_threshold) or (hypo.seq_length > self.runlength_threshold):
            self.logger.log(logging.DEBUG, f'monitor_hypothesis: cum bayes fac {hypo.cum_bayes_fac} < {self.bayes_factor_threshold} or seq length {hypo.seq_length} > {self.runlength_threshold}, potential outlier')
            return False
        self.logger.log(logging.DEBUG, 'monitor_hypothesis: no intervention required')
        return True

    def reset_monitor(self, hypo_index: int):
        """
        Clear the monitoring data from the provided hypothesis

        Parameters
        ----------
        hypo_index
            The index of the hypothesis we want to clear the monitor data from
        """

        try:
            hypo = self.hypotheses[hypo_index]
        except IndexError:
            self.logger.log(logging.ERROR, f'monitor_hypothesis: Unable to pull hypothesis at index {hypo_index}')
            return False
        hypo.cum_bayes_fac = 1.0
        hypo.seq_length = 0
        self.logger.log(logging.DEBUG, 'reset_monitor: clear the monitoring data from the provided hypothesis')

    def update_hypothesis(self, hypo_index: int, depth: float, variance: float):
        """
        Update the given hypothesis (index is provided) being tracked at this node.  This implements the standard
        univariate dynamic linear model update equations (West & Harrison, 'Bayesian Forecasting and Dynamic Models',
        Springer, 2ed, 1997, Ch. 2), along with the Bayes factor monitoring code (W&H, Ch. 11).  The only failure mode
        possible with this code is if the input data would cause an intervention to be requested on the current track.
        In this case, it is the caller's responsibility to utilise the data point, since it will not be incorporated
        into the hypothesis --- typically this would mean adding a new hypothesis and pushing it onto the stack.

        Parameters
        ----------
        hypo_index
            The index of the hypothesis we want to update
        depth
            estimate of depth
        variance
            estimate of variance

        Returns
        -------
        bool
            Returns False if the estimate does not really match the track that the hypothesis represents (i.e., an
            intervention is required).
        """

        hypo = self.hypotheses[hypo_index]
        # check current estimate with node monitoring
        monitoring_answer = self.monitor_hypothesis(hypo_index, depth, variance)
        if not monitoring_answer:
            self.logger.log(logging.DEBUG, 'update_hypothesis: monitoring determined an intervention is required')
            return False

        if self.variance_selection != 'cube':
            hypo.variance_estimate = (hypo.number_of_points - 1) * hypo.variance_estimate / hypo.number_of_points + (depth - hypo.current_depth) ** 2 / hypo.number_of_points
        # add capability to 'age' the sounding with a discount factor.
        sys_variance = hypo.current_variance * (1.0 - self.discount) / self.discount

        gain = hypo.predict_variance / (variance + hypo.predict_variance)
        innovation = depth - hypo.predict_depth
        hypo.predict_depth += gain * innovation
        hypo.current_depth = hypo.predict_depth
        hypo.current_variance = variance * hypo.predict_variance / (variance + hypo.predict_variance)
        hypo.predict_variance = hypo.current_variance + sys_variance
        hypo.number_of_points += 1
        self.logger.log(logging.DEBUG, f'update_hypothesis: hypothesis number {hypo_index} updated with depth {depth} and variance {variance}')
        return True

    def best_hypothesis_index(self, depth: float, variance: float):
        """
        Find the closest matching hypothesis in the current hypothesis list.  This computes the normalized absolute error
        between one-step forecast for each hypothesis currently being tracked and the input sample, and returns the index
        of the hypothesis with the smallest error value.  If there is more than one node with the same error (unlikely
        in practice, but possible), then the last one in the list is chosen.

        Parameters
        ----------
        depth
            current input sample to be matched
        variance
            current input variance to be matched

        Returns
        -------
        int
            index to the best hypothesis
        """

        best_hypo_index = None
        min_error = None
        for idx, hyp in enumerate(self.hypotheses):
            forecast_variance = hyp.predict_variance + variance
            error = abs((depth - hyp.predict_depth) / np.sqrt(forecast_variance))
            if (min_error and error < min_error) or (min_error is None):
                min_error = error
                best_hypo_index = idx
                self.logger.log(logging.DEBUG, f'best_hypothesis_index: hypothesis number {best_hypo_index} picked with minimum error {min_error}')
        return best_hypo_index

    def choose_hypothesis(self):
        """
        Choose the best hypothesis for this node.  In this context, `best' means `hypothesis with most points',
        rather than through any other metric.  This may not be the `best' until all of the data is in, but it should
        give an idea of what's going on in the data structure at any point (particularly if it changes dramatically
        from sample to sample)

        Returns
        -------
        Hypothesis
            the hypothesis with the most points
        float
            the hypothesis strength ratio for this hypothesis (how convinced CUBE is that the hypo is good)

        """
        best_hypo = None
        hypo_ratio = 0.0
        second_highest_count = 0
        current_max_pointcount = 0
        for hyp in self.hypotheses:
            if hyp.number_of_points > 0:
                if hyp.number_of_points > current_max_pointcount:
                    best_hypo = hyp
                    if current_max_pointcount > second_highest_count:
                        second_highest_count = current_max_pointcount
                    current_max_pointcount = hyp.number_of_points
                elif hyp.number_of_points > second_highest_count:
                    second_highest_count = hyp.number_of_points
        if second_highest_count and current_max_pointcount:
            hypo_ratio = max(0.0, self.max_hypothesis_ratio - (current_max_pointcount / second_highest_count))
        self.logger.log(logging.DEBUG, f'choose_hypothesis: hypothesis number {best_hypo.hypothesis_number} picked as it had the most points ({current_max_pointcount}), hypothesis strength {hypo_ratio}')
        return best_hypo, hypo_ratio

    def update_node(self, depth: float, variance: float):
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
        depth
            new depth estimate to incorporate
        variance
            new variance estimate to incorporate

        Returns
        -------
        bool
            True if node hypothesis was updated or if this is the first update.  False if a new hypothesis had to be
            created because data did not allow updating an existing one.
        """
        # find the best matching hypothesis index for the current input sample given those currently being tracked
        best_idx = self.best_hypothesis_index(depth, variance)
        if best_idx is None:
            # didn't match one, should only happen where there are no hypothesis, so we add a new one
            self.add_hypothesis(depth, variance, null_hypothesis=False)
        else:
            # update the best hypothesis with the current data
            updated = self.update_hypothesis(best_idx, depth, variance)
            if not updated:
                # failed update - indicates an intervention, so that we need to start a new hypothesis to capture the outlier/datum shift
                self.reset_monitor(best_idx)
                self.add_hypothesis(depth, variance, null_hypothesis=False)
                self.logger.log(logging.DEBUG, f'update_node: no hypothesis updated, depth {depth} variance {variance} successfully incorporated as new hypothesis')
        return True

    def truncate(self):
        """
        Identify all points that are outliers and remove them from the queue.  The definition of 'outlier' depends on
        the self.quotient_limit attribute.  In general, the higher the value, the more extreme the difference between
        mean depth and point depth must be to be considered an outlier.

        In theory, the distribution of the quotient values computed should be approximately a Fisher F(1,N-2) where
        there are N points in the input sequence.  The values of the quotients are always positive, and monotonically
        increasing for worse outliers; therefore, one-sided critical values should be considered.
        """

        if self.n_queued < 3:
            self.logger.log(logging.DEBUG, f'truncate: with {self.n_queued} queued points, truncate unnecessary')
            return
        mean = 0.0
        sum_square_diff = 0.0
        num_points = self.n_queued - 1  # number of points + 1 outlier
        for depth, variance in self.queue:
            mean += depth
            sum_square_diff += depth ** 2
        sum_square_diff -= (mean ** 2) / (num_points + 1)
        mean /= (num_points + 1)
        sum_square_diff_k = num_points * sum_square_diff / (num_points ** 2 - 1)

        # run the list computing quotients, gather indicies for the outliers
        outlier_index = []
        for idx in range(len(self.queue)):
            depth = self.queue[idx][0]
            diff_sq = (depth - mean) ** 2
            quot = diff_sq / (sum_square_diff_k - (diff_sq / (num_points - 1)))
            if quot >= self.quotient_limit:
                self.logger.log(logging.DEBUG, f'truncate: point {idx} flagged for removal with quotient value greater than limit {self.quotient_limit}')
                outlier_index.append(idx)
        outlier_index = outlier_index[::-1]
        # now remove the marked entities from the queue
        for idx in outlier_index:
            self.queue.pop(idx)
            self.n_queued -= 1
        self.logger.log(logging.DEBUG, f'truncate: removed {len(outlier_index)} points from the queue')

    def flush_queue(self):
        """
        This flushes the queue into the input sequence in order (i.e., take current median, resort, repeat).  Since the
        queue is always sorted, we can just walk the list in order, rather than having to re-sort or shift data, etc.
        When we have an even number of points, we take the shallowest of the points first; this means that we walk the
        list alternately to the left and right, starting to the right if the initial number of points is even, and to
        the left if the number of points is odd.  To avoid shifting the data, we just increase the step after every
        extraction, until we step off the LHS of the array.

        For a list of 10 points, the order ends up looking something like this: [4, 5, 3, 6, 2, 7, 1, 8, 0, 9]
        """

        if self.n_queued == 0:
            self.logger.log(logging.DEBUG, f'flush_queue: no queued points to flush')
            return
        scale = 1
        self.truncate()
        self.logger.log(logging.DEBUG, f'flush_queue: flushing {self.n_queued} points')
        if self.n_queued % 2 == 0:  # even
            ex_pt = int(self.n_queued / 2 - 1)
            direction = 1
        else:  # odd
            ex_pt = int(self.n_queued / 2)
            direction = -1
        while ex_pt >= 0:
            self.update_node(self.queue[ex_pt][0], self.queue[ex_pt][1])
            ex_pt += direction * scale
            direction = -direction
            scale += 1
        self.queue = []
        self.n_queued = 0

    def queue_fill(self, depth: float, variance: float):
        """
        Insert a new point into the queue, maintain depth sorted order, with greater depths last

        Parameters
        ----------
        depth
            new depth value to add to the queue
        variance
            new variance value to add to the queue
        """

        if not self.use_queue:
            self.logger.log(logging.WARNING, f'queue_fill: skipping as use_queue is False')
            return
        if not self.queue:
            self.logger.log(logging.DEBUG, f'queue_fill: queue is empty, adding depth {depth}, variance {variance} to queue')
            self.queue.append([depth, variance])
        else:
            insertion_index = None
            for i in range(len(self.queue)):
                q_dpth = self.queue[i][0]
                if depth > q_dpth:
                    insertion_index = i + 1
                else:
                    break
            if insertion_index is not None:
                self.logger.log(logging.DEBUG, f'queue_fill: inserted at index {insertion_index}, adding depth {depth}, variance {variance} to queue')
                self.queue.insert(insertion_index, [depth, variance])
            else:
                self.logger.log(logging.DEBUG, f'queue_fill: adding to end of queue, adding depth {depth}, variance {variance} to queue')
                self.queue.insert(0, [depth, variance])
        self.n_queued += 1

    def queue_insert(self, depth: float, variance: float):
        """
        Insert a point in the already filled queue.  We then return the median point and insert this new point, ensuring
        that the queue remains sorted, with greater depths first.

        Parameters
        ----------
        depth
            new depth value to add to the queue
        variance
            new variance value to add to the queue

        Returns
        -------
        float
            median depth
        float
            median variance
        """

        if not self.use_queue:
            self.logger.log(logging.WARNING, f'queue_insert: skipping as use_queue is False')
            return depth, variance
        # 11 / 2 = 5.5, floor(5.5) = 5, 5 being the median index of an array of 11 points
        median_index = int(np.floor(self.median_length / 2))
        mdepth, mvariance = self.queue.pop(median_index)
        insertion_index = None
        if depth > mdepth:
            check_range = range(median_index, len(self.queue))
        else:
            check_range = range(len(self.queue))

        for i in check_range:
            q_dpth = self.queue[i][0]
            if depth < q_dpth:
                insertion_index = i
                break

        if insertion_index is not None:
            self.logger.log(logging.DEBUG, f'queue_insert: queue full, inserted at index {insertion_index}, adding depth {depth}, variance {variance} to queue')
            self.queue.insert(insertion_index, [depth, variance])
        else:
            self.logger.log(logging.DEBUG, f'queue_insert: queue full, adding to end of queue, adding depth {depth}, variance {variance} to queue')
            self.queue.append([depth, variance])

        # compute the likely 99% confidence bound below the shallowest point and above the deepest point in the
        # buffer, and check that they do actually overlap somewhere in the middle.  Otherwise, with less than 1% chance
        # of error, we are suspicious that there are outliers in the buffer somewhere and we should attempt a round of
        # outlier rejection.  Assuming that the errors are approximately normal, 0.5% in either tail is achieved at
        # 2.5758 std dev from the mean.

        # 2.56 being the 99 percent confidence interval for normal distribution
        first_pt = self.queue[0]
        last_pt = self.queue[-1]
        low_water = last_pt[0] - 2.56 * np.sqrt(last_pt[1])
        high_water = first_pt[0] + 2.56 * np.sqrt(first_pt[1])
        if low_water >= high_water:  # confidence limits do not overlap
            self.truncate()  # remove any outliers
        self.logger.log(logging.DEBUG, f'queue_insert: queue full, returning median point depth {mdepth}, variance {mvariance}')
        return mdepth, mvariance

    def add_to_queue(self, depth: float, variance: float):
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
        depth
            new depth value to add to the queue
        variance
            new variance value to add to the queue
        """

        if not self.queue:
            self.n_queued = 0
        if self.use_queue:
            self.logger.log(logging.DEBUG, f'add_to_queue: adding depth {depth} variance {variance} to the queue')
            if self.n_queued < self.median_length:
                self.queue_fill(depth, variance)
            else:
                median_depth, median_variance = self.queue_insert(depth, variance)
                self.update_node(median_depth, median_variance)
        else:
            self.update_node(depth, variance)

    def add_point_to_node(self, depth: float, vertical_uncertainty: float, horizontal_uncertainty: float, distance_to_node: float):
        """
        Insert a point into the node.  This will compute the variance scale factor for the new data, and send the data
        into the estimation queue.

        Parameters
        ----------
        depth
            new depth value to add to the queue
        vertical_uncertainty
            new vertical uncertainty value associated with the point, assumes 2 sigma
        horizontal_uncertainty
            new horizontal uncertainty value associated with the point, assumes 2 sigma
        distance_to_node
            distance from point to node
        """

        conf_95_percent = 1.96
        if np.isnan(self.predicted_depth):
            self.logger.log(logging.DEBUG, f'add_point_to_node: Sounding rejected with predicted depth of NaN, sounding depth = {depth}')
            return
        # euclidean distance in projected space, i.e. distance sounding is being propagated from touchdown boresight
        #   to node estimation point
        dist = np.sqrt(distance_to_node)
        if self.predicted_depth:
            target_depth = self.predicted_depth
            # do the test for blunders here, since it makes no sense to test when there is no predicted depth
            # blunders = beam solutions generated by the multibeam that do not correctly represent the seafloor
            blunder_limit = min(target_depth - self.blunder_min, target_depth - self.blunder_percent * abs(target_depth))
            blunder_limit = min(blunder_limit, target_depth - self.blunder_scalar * np.sqrt(self.predicted_variance))
            if depth < blunder_limit:
                self.logger.log(logging.DEBUG, f'add_point_to_node: Sounding rejected, {depth} less than blunder limit {blunder_limit}')
                return
        else:
            self.logger.log(logging.DEBUG, f'add_point_to_node: Blunder limit test pass, no predicted depth for this node')
            target_depth = depth
        calculated_captdist = self.capture_dist_scale * abs(target_depth)
        if dist > max(calculated_captdist, 0.5):
            self.logger.log(logging.DEBUG, f'add_point_to_node: sounding rejected, {dist} greater than max(0.5 or calculated capture distance {calculated_captdist})')
            return
        self.logger.log(logging.DEBUG, f'add_point_to_node: sounding accepted at node, distance {dist}m, target depth {max(calculated_captdist, 0.5)}m')
        # add horizontal positioning uncertainty, assumes 2sigma
        dist += conf_95_percent * np.sqrt(horizontal_uncertainty)
        # TODO this asked for range (range != 0) in the original source, don't have range
        sounding_range = 0.0
        if sounding_range != 0.0 and (not np.isnan(self.predicted_depth) and self.predicted_depth):
            offset = self.predicted_depth - depth
            self.logger.log(logging.DEBUG, f'add_point_to_node: adding offset to depth')
        else:
            offset = 0.0
        variance = vertical_uncertainty * (1.0 + self.var_scale * (dist ** self.dist_exponent))
        self.add_to_queue(depth + offset, variance)
        self.nominated = None

    def _return_nominated_answer(self, value: tuple = ('depth', 'uncertainty')):
        """
        Return the data for the provided value identifiers for the nominated hypothesis

        Parameters
        ----------
        value
            list of the values you want to extract from this node, one of 'depth', 'uncertainty', 'ratio', 'n_hypotheses'

        Returns
        -------
        list
            list of floats for each value identifier provided
        """

        data = []
        for value_id in value:
            if value_id == 'depth':
                data.append(self.nominated.current_depth)
            elif value_id == 'uncertainty':
                data.append(self.stddev_to_conf_scale * np.sqrt(self.nominated.current_variance))
            elif value_id == 'ratio':
                data.append(0.0)  # having a nominated hypothesis means there is no ratio, only one hypothesis
            elif value_id == 'n_hypotheses':
                data.append(self.return_number_of_hypotheses())
        self.logger.log(logging.DEBUG, f'_return_nominated_answer: good hypothesis, returning {data} for {value}')
        return data

    def _return_answer_from_hypothesis(self, hyp: Hypothesis, ratio: float, value: tuple = ('depth', 'uncertainty')):
        """
        Provide the answer for the given value identifiers for the given hypothesis.

        Parameters
        ----------
        hyp
            selected hypothesis to get an answer from
        ratio
            hypothesis strength ratio for this hypothesis
        value
            list of the values you want to extract from this node, one of 'depth', 'uncertainty', 'ratio', 'n_hypotheses'

        Returns
        -------
        list
            list of floats for each value identifier provided
        """

        data = []
        if hyp.number_of_points > 0:
            # Only reconstruct if some data was involved in the construction of the hypothesis.  This excludes
            # initial hypotheses from an initialisation surface, which are set up with n_j = 0.
            for value_id in value:
                if value_id == 'depth':
                    data.append(hyp.current_depth)
                elif value_id == 'uncertainty':
                    if self.variance_selection == 'max':
                        data.append(self.stddev_to_conf_scale * np.sqrt(max(hyp.current_variance, hyp.variance_estimate)))
                    elif self.variance_selection == 'input':
                        data.append(self.stddev_to_conf_scale * np.sqrt(hyp.variance_estimate))
                    else:
                        data.append(self.stddev_to_conf_scale * np.sqrt(hyp.current_variance))
                elif value_id == 'ratio':
                    data.append(ratio)  # only one hypothesis
                elif value_id == 'n_hypotheses':
                    data.append(self.return_number_of_hypotheses())
            self.logger.log(logging.DEBUG, f'_return_answer_from_hypothesis: good hypothesis, returning {data} for {value}')
        else:
            self.logger.log(logging.DEBUG, f'_return_answer_from_hypothesis: hypothesis empty, returning nodatavalues')
            for _ in value:
                data.append(self.no_data_value)
        return data

    def extract_node_value(self, value: tuple = ('depth', 'uncertainty')):
        """
        Extract a node value for each of the provided value identifiers.  These values come from either the nominated
        hypothesis or a selected hypothesis that has the most points of all hypotheses.  If there is no hypothesis,
        you will get a self.no_data_value for each requested value identifier.

        Parameters
        ----------
        value
            list of the values you want to extract from this node, one of 'depth', 'uncertainty', 'ratio', 'n_hypotheses'

        Returns
        -------
        list
            list of the values for each value identifier provided
        """

        self.logger.log(logging.DEBUG, f'extract_node_value: getting hypothesis answer for {value}')
        if self.nominated is not None:
            self.logger.log(logging.DEBUG, f'extract_node_value: using nominated hypothesis')
            return self._return_nominated_answer(value)
        if not self.hypotheses:
            self.logger.log(logging.DEBUG, f'extract_node_value: no hypothesis found')
            data = []
            for _ in value:
                data.append(self.no_data_value)
        elif len(self.hypotheses) == 1:  # Special case: only one depth hypothesis (the usual case, we hope ...)
            self.logger.log(logging.DEBUG, f'extract_node_value: only one hypothesis!')
            hyp = self.hypotheses[0]
            data = self._return_answer_from_hypothesis(hyp, 0.0, value)  # ratio of 0 for only having one hypothesis
        else:
            self.logger.log(logging.DEBUG, f'extract_node_value: multiple hypotheses found')
            hyp, ratio = self.choose_hypothesis()
            data = self._return_answer_from_hypothesis(hyp, ratio, value)
        return data

    def extract_closest_node_value(self, depth: float, variance: float, value: tuple = ('depth', 'uncertainty')):
        """
        Extract the node values for the hypothesis which is closest in depth to the supplied depth/variance point values, in a
        minimum error sense.  If there are no depth hypotheses in this node, self.no_data_value is returned.

        Parameters
        ----------
        depth
            depth value to use in the query
        variance
            variance value to use in the query
        value
            list of the values you want to extract from this node, one of 'depth', 'uncertainty', 'ratio', 'n_hypotheses'

        Returns
        -------
        list
            list of the values for each value identifier provided
        """

        if self.nominated is not None:
            self.logger.log(logging.DEBUG, f'extract_closest_node_value: using nominated hypothesis')
            return self._return_nominated_answer(value)

        if not self.hypotheses or len(self.hypotheses) == 1:
            self.logger.log(logging.DEBUG, f'extract_closest_node_value: one hypothesis or less, falling back on basic extraction')
            # with no hypotheses or just one hypothesis, this is just doing the basic extraction
            return self.extract_node_value(value)
        min_error = None
        nearest_hypo = None
        total_points = 0
        for hyp in self.hypotheses:
            if hyp.number_of_points > 0:  # check that some data were used in making the hypothesis before accepting it as valid
                error = abs((hyp.current_depth - depth) / np.sqrt(variance))
                if min_error is None or error < min_error:
                    min_error = error
                    nearest_hypo = hyp
                total_points += hyp.number_of_points
        if nearest_hypo is None:  # should never get to this point
            self.logger.log(logging.WARNING, f'extract_closest_node_value: no hypothesis found!')
            data = []
            for _ in value:
                data.append(self.no_data_value)
        else:
            ratio = max(0.0, self.max_hypothesis_ratio - (nearest_hypo.number_of_points / (total_points - nearest_hypo.number_of_points)))
            data = self._return_answer_from_hypothesis(nearest_hypo, ratio, value)
            self.logger.log(logging.DEBUG, f'extract_closest_node_value: found {data} for {value}')
        return data

    def extract_posterior_weighted_node_value(self, depth: float, variance: float, value: tuple = ('depth', 'uncertainty')):
        """
        Extract a posterior weighted best depth hypothesis using the provided depth/variance values as a guide.  Returns
        the hypothesis value for each provided value identifier in the list.

        Parameters
        ----------
        depth
            depth value to use in the query
        variance
            variance value to use in the query
        value
            list of the values you want to extract from this node, one of 'depth', 'uncertainty', 'ratio', 'n_hypotheses'

        Returns
        -------
        list
            list of the values for each value identifier provided
        """

        if self.nominated is not None:
            self.logger.log(logging.DEBUG, f'extract_posterior_weighted_node_value: using nominated hypothesis')
            return self._return_nominated_answer(value)
        if not self.hypotheses or len(self.hypotheses) == 1:
            self.logger.log(logging.DEBUG, f'extract_posterior_weighted_node_value: one hypothesis or less, falling back on basic extraction')
            # with no hypotheses or just one hypothesis, this is just doing the basic extraction
            return self.extract_node_value(value)
        max_posterior = None
        nearest_hypo = None
        total_points = 0
        for hyp in self.hypotheses:
            if hyp.number_of_points > 0:  # check that some data were used in making the hypothesis before accepting it as valid
                mean = hyp.current_depth
                posterior = -(depth - mean) ** 2 / (2.0 * variance) + np.log(hyp.number_of_points)
                if max_posterior is None or posterior > max_posterior:
                    max_posterior = posterior
                    nearest_hypo = hyp
                total_points += hyp.number_of_points
        if nearest_hypo is None:  # should never get to this point
            self.logger.log(logging.WARNING, f'extract_posterior_weighted_node_value: no hypothesis found!')
            data = []
            for _ in value:
                data.append(self.no_data_value)
        else:
            ratio = max(0.0, self.max_hypothesis_ratio - (nearest_hypo.number_of_points / (total_points - nearest_hypo.number_of_points)))
            data = self._return_answer_from_hypothesis(nearest_hypo, ratio, value)
            self.logger.log(logging.DEBUG, f'extract_posterior_weighted_node_value: found {data} for {value}')
        return data

    def return_depth(self):
        """
        Return the depth for the 'best' hypothesis in this node.  In this case, the 'best' hypothesis is the hypothesis
        with the most points.  If there is no hypothesis, this returns self.no_data_value

        Returns
        -------
        float
            depth value for the best hypothesis
        """

        return self.extract_node_value(('depth', ))[0]

    def return_uncertainty(self):
        """
        Return the uncertainty for the 'best' hypothesis in this node.  In this case, the 'best' hypothesis is the hypothesis
        with the most points.  If there is no hypothesis, this returns self.no_data_value

        Returns
        -------
        float
            uncertainty value for the best hypothesis
        """

        return self.extract_node_value(('uncertainty', ))[0]

    def return_number_of_hypotheses(self):
        """
        Return the total number of hypotheses in this node.

        Returns
        -------
        int
            the total number of hypotheses in the node
        """

        return len(self.hypotheses)

    def dump_hypotheses(self):
        """
        Print the status of each hypothesis
        """

        for hyp in self.hypotheses:
            print('Hypothesis {} - depth={}, variance={}, number_of_points={}'.format(hyp.hypothesis_number, hyp.current_depth,
                                                                                      hyp.current_variance, hyp.number_of_points))


class CubeGrid:
    def __init__(self, minimum_easting: float, maximum_northing: float, num_columns: int, num_rows: int,
                 resolution_x: float, resolution_y: float, param: CubeParameters, use_queue: bool = True,
                 logfile: str = None, debug: bool = False):
        """
        Main structure for Cube, holds CubeNodes in a grid with metadata.

        Parameters
        ----------
        minimum_easting
            minimum easting extent of the grid
        maximum_northing
            maximum northing extent of the grid
        num_columns
            number of columns in the grid
        num_rows
            number of rows in the grid
        resolution_x
            the resolution in the x direction of the grid (width of columns)
        resolution_y
            the resolution in the y direction of the grid (height of rows)
        param
            CubeParameters object with the default settings for the grid
        use_queue
            Executes the 'Reordering' step, see CUBE User Manual 3.1.  With this set to False, this step is skipped.
            User Manual states that with multiple hypothesis implementation of CUBE, Reordering is no longer necessary.
        logfile
            optional path to a logfile
        debug
            if True, will print debug messages to the logger
        """

        self.param = param
        self.no_data_value = param.no_data_value
        self.dist_scale = param.dist_scale
        self.inv_dist_exponent = param.inv_dist_exponent
        self.min_context = param.min_context
        self.max_context = param.max_context
        self.iho_order = param.iho_order
        self.iho_fixed, self.iho_percent = get_iho_limits(param.iho_order)
        self.use_queue = use_queue

        self.minimum_easting = minimum_easting
        self.maximum_northing = maximum_northing

        self.num_columns = num_columns
        self.num_rows = num_rows
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        self.logfile = logfile

        loglvl = logging.INFO
        if debug:
            loglvl = logging.DEBUG

        if self.logfile:
            self.logger = return_logger(self.logfile, loglevel=loglvl)
        else:
            self.logger = return_logger(loglevel=loglvl)
        self.debug = debug

        self.grid = []
        for row in range(self.num_rows):
            rdata = []
            for column in range(self.num_columns):
                rdata.append(CubeNode(bayes_factor_threshold=param.bayes_factor_threshold, est_offset=param.est_offset,
                                      runlength_threshold=param.runlength_threshold, discount=param.discount, quotient_limit=param.quotient_limit,
                                      median_length=param.median_length, blunder_min=param.blunder_min, blunder_percent=param.blunder_percent,
                                      blunder_scalar=param.blunder_scalar, capture_dist_scale=param.capture_dist_scale,
                                      var_scale=param.var_scale, dist_exponent=param.dist_exponent, stddev_to_conf_scale=param.stddev_to_conf_scale,
                                      no_data_value=param.no_data_value, variance_selection=param.variance_selection, use_queue=use_queue, logger=self.logger))
            self.grid.append(rdata)

    # def __repr__(self):
    #     print(f'CubeGrid: (rows={self.num_rows} x columns={self.num_columns})')
    #     print('**************************************')
    #     print(f'Contains {self.total_nodes_count} total nodes, {self.empty_nodes_count} empty and {self.populated_nodes_count} populated')

    def _validate_insert_points(self, depth: np.ndarray, horizontal_uncertainty: np.ndarray, vertical_uncertainty: np.ndarray,
                                easting: np.ndarray, northing: np.ndarray):
        if isinstance(depth, (int, float)):
            depth = np.array([depth])
        elif isinstance(depth, list):
            depth = np.array(depth)

        if isinstance(horizontal_uncertainty, (int, float)):
            horizontal_uncertainty = np.array([horizontal_uncertainty])
        elif isinstance(horizontal_uncertainty, list):
            horizontal_uncertainty = np.array(horizontal_uncertainty)

        if isinstance(vertical_uncertainty, (int, float)):
            vertical_uncertainty = np.array([vertical_uncertainty])
        elif isinstance(vertical_uncertainty, list):
            vertical_uncertainty = np.array(vertical_uncertainty)

        if isinstance(easting, (int, float)):
            easting = np.array([easting])
        elif isinstance(easting, list):
            easting = np.array(easting)

        if isinstance(northing, (int, float)):
            northing = np.array([northing])
        elif isinstance(northing, list):
            northing = np.array(northing)

        assert depth.size == horizontal_uncertainty.size == vertical_uncertainty.size == easting.size == northing.size
        return depth, horizontal_uncertainty, vertical_uncertainty, easting, northing

    @property
    def populated_nodes_count(self):
        cnt = 0
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                node = self.grid[row][col]
                if node.hypotheses or node.n_queued:
                    cnt += 1
        return cnt

    @property
    def empty_nodes_count(self):
        cnt = 0
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                node = self.grid[row][col]
                if not node.hypotheses and not node.n_queued:
                    cnt += 1
        return cnt

    @property
    def total_nodes_count(self):
        return self.num_rows * self.num_columns

    def insert_points(self, depth: np.ndarray, horizontal_uncertainty: np.ndarray, vertical_uncertainty: np.ndarray,
                      easting: np.ndarray, northing: np.ndarray):
        """
        Add an array of point values to the grid

        Parameters
        ----------
        depth
            new depth values to add to the queue
        vertical_uncertainty
            new vertical uncertainty values associated with the points, assumes 2 sigma
        horizontal_uncertainty
            new horizontal uncertainty values associated with the points, assumes 2 sigma
        easting
            new easting values associated with the points
        northing
            new northing values associated with the points
        """

        depth, horizontal_uncertainty, vertical_uncertainty, easting, northing = self._validate_insert_points(depth, horizontal_uncertainty, vertical_uncertainty, easting, northing)
        conf_95_percent = 1.96
        conf_99_percent = 2.95
        self.logger.log(logging.DEBUG, f'insert_points: Adding {len(depth)} points...')
        for i in range(len(depth)):
            self.logger.log(logging.DEBUG, f'insert_points: x:{easting[i]}, y:{northing[i]}, z:{depth[i]}, thu:{horizontal_uncertainty[i]}, tvu:{vertical_uncertainty[i]}')
            # Determine IHO S-44 derived limits on maximum variance
            max_variance_allowed = (self.iho_fixed + self.iho_percent * depth[i] ** 2) / conf_95_percent ** 2
            ratio = max_variance_allowed / vertical_uncertainty[i]
            if ratio <= 2.0:
                ratio = 2.0
            max_radius = conf_99_percent * np.sqrt(horizontal_uncertainty[i])
            radius = self.dist_scale * (ratio - 1.0) ** self.inv_dist_exponent - max_radius
            if radius < 0.0:
                radius = self.dist_scale
            elif radius > max_radius:
                radius = max_radius
            if radius < self.dist_scale:
                radius = self.dist_scale
            self.logger.log(logging.DEBUG, f'insert_points: dist_scale:{self.dist_scale}, ratio:{ratio}, max radius:{max_radius}, max_variance:{max_variance_allowed}')
            # determine the coordinates of the effect square.  This is designed to compute the largest region the sounding
            # can effect, and hence to make the insertion more efficient by only offering the sounding where it is likely
            # to be used
            min_x = int(((easting[i] - radius) - self.minimum_easting) / self.resolution_x)
            max_x = int(((easting[i] + radius) - self.minimum_easting) / self.resolution_x)
            min_y = int((self.maximum_northing - (northing[i] + radius)) / self.resolution_y)
            max_y = int((self.maximum_northing - (northing[i] - radius)) / self.resolution_y)
            # check that the sounding hits somewhere in the grid
            if max_x < 0 or min_x >= (self.num_columns - 1) or max_y < 0 or min_y >= (self.num_rows - 1):
                self.logger.log(logging.DEBUG, f'insert_points: Sounding out of bounds, ({min_x},{min_y}) ({max_x},{max_y})')
                continue  # out of bounds
            # clip to the interior of the current grid
            min_x = max(0, min_x)
            max_x = min(max_x, self.num_columns - 1)
            min_y = max(0, min_y)
            max_y = min(max_y, self.num_rows - 1)
            self.logger.log(logging.DEBUG, f'insert_points: clipped row,column limits to use in search, ({min_x},{min_y}) ({max_x},{max_y})')
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    node_x = self.minimum_easting + x * self.resolution_x
                    node_y = self.maximum_northing - y * self.resolution_y
                    distance_sq = (node_x - easting[i]) ** 2 + (node_y - northing[i]) ** 2
                    if distance_sq >= radius ** 2:
                        self.logger.log(logging.DEBUG, f'insert_points: rejecting point as out of distance to node at row/col, ({y}, {x})')
                        continue  # distance to great, not including this point in this node
                    self.logger.log(logging.DEBUG, f'insert_points: adding point to node at row/col, ({y}, {x})')
                    self.grid[y][x].add_point_to_node(depth[i], vertical_uncertainty[i], horizontal_uncertainty[i], distance_sq)

    def flush_node_queues(self):
        """
        Flush the queues for each node
        """
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                node = self.grid[row][col]
                node.flush_queue()

    def get_grid_values(self, value: tuple = ('depth', 'uncertainty'), method: str = 'local'):
        """
        Get the values for each node in the grid for each value identifier in the value list.

        Parameters
        ----------
        value
            list of the values you want to extract from this grid, one of 'depth', 'uncertainty', 'ratio', 'n_hypotheses'
        method
            method to use in determining the appropriate hypothesis value.  'local' to use the local spatial
            context to find the closest node with a single hypothesis and use that hypothesis depth to find the nearest
            hypothesis in terms of depth in the current node.  'prior' to use the hypothesis with the most points
            associated with it.  'posterior' to combine both prior and local methods to form an approximate Bayesian
            posterior distribution.  'predict' to get the hypothesis closest to the predicted depth associated with
            each node.

        Returns
        -------
        list
            list of numpy arrays for the grid, one for each variable identifier in the provided value list
        """
        data = []
        self.logger.log(logging.DEBUG, f'get_grid_values: getting grid values for {value} using method {method}')
        for _ in value:
            vgrid = np.full((self.num_rows, self.num_columns), self.no_data_value)
            data.append(vgrid)
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                node = self.grid[row][col]
                node_data = data
                if method in ['local', 'posterior']:
                    node_hcount = node.return_number_of_hypotheses()
                    if node_hcount <= 1:
                        node_data = node.extract_node_value(value)
                    else:
                        closest_node = None
                        for offset in range(self.min_context, self.max_context + 1):
                            target_rows = [row - offset, row + offset]
                            for target_row in target_rows:
                                if 0 <= target_row < self.num_rows:
                                    for col_offset in range(-offset, offset + 1):
                                        target_col = col + col_offset
                                        if target_col < 0 or target_col >= self.num_columns:
                                            continue
                                        chk_node_hcount = self.grid[target_row][target_col].return_number_of_hypotheses()
                                        if chk_node_hcount == 1:
                                            closest_node = self.grid[target_row][target_col]
                                            self.logger.log(logging.DEBUG, f'get_grid_values: found closest node during row search at ({target_row},{target_col})')
                                            break
                                if closest_node is not None:
                                    break
                            if closest_node is None:
                                target_cols = [col - offset, col + offset]
                                for target_col in target_cols:
                                    if 0 <= target_col < self.num_columns:
                                        for row_offset in range(-offset + 1, offset):
                                            target_row = row + row_offset
                                            if target_row < 0 or target_row >= self.num_rows:
                                                continue
                                            chk_node_hcount = self.grid[target_row][target_col].return_number_of_hypotheses()
                                            if chk_node_hcount == 1:
                                                closest_node = self.grid[target_row][target_col]
                                                self.logger.log(logging.DEBUG, f'get_grid_values: found closest node during column search at ({target_row},{target_col})')
                                                break
                                    if closest_node is not None:
                                        break
                            if closest_node is not None:
                                break
                        if closest_node is None:  # default to the basic node hypothesis selection, couldn't find a good hypothesis in the region
                            self.logger.log(logging.DEBUG, f"get_grid_values: default to the basic node hypothesis selection, couldn't find a good hypothesis in the region")
                            node_data = node.extract_node_value(value)
                        else:
                            self.logger.log(logging.DEBUG, f"get_grid_values: extract value from closest node found")
                            closest_data = closest_node.extract_node_value(('depth', 'uncertainty'))
                            if method == 'local':
                                node_data = node.extract_closest_node_value(closest_data[0], closest_data[1], value)
                            elif method == 'posterior':
                                node_data = node.extract_posterior_weighted_node_value(closest_data[0], closest_data[1], value)
                elif method == 'prior':
                    node_data = node.extract_node_value(value)
                elif method == 'predicted':
                    node_data = node.extract_closest_node_value(node.predicted_depth, node.predicted_variance, value)
                for cnt, node_value in enumerate(node_data):
                    data[cnt][row, col] = node_value
        return data

    def get_grid_depth(self, method: str = 'local'):
        """
        Shortcut for get_grid_values, if you are only interested in depth.

        Parameters
        ----------
        method
            one of 'local', 'posterior', 'prior', 'predicted'.  See get_grid_values for more info.

        Returns
        -------
        np.ndarray
            2d numpy array of node depth values
        """

        if method == 'local':
            return self.get_grid_values(('depth',), 'local')[0]
        elif method == 'posterior':
            return self.get_grid_values(('depth',), 'posterior')[0]
        elif method == 'prior':
            return self.get_grid_values(('depth',), 'prior')[0]
        elif method == 'predicted':
            return self.get_grid_values(('depth',), 'predicted')[0]

    def get_grid_uncertainty(self, method: str = 'local'):
        """
        Shortcut for get_grid_values, if you are only interested in uncertainty.

        Parameters
        ----------
        method
            one of 'local', 'posterior', 'prior', 'predicted'.  See get_grid_values for more info.

        Returns
        -------
        np.ndarray
            2d numpy array of node uncertainty values
        """

        if method == 'local':
            return self.get_grid_values(('uncertainty',), 'local')[0]
        elif method == 'posterior':
            return self.get_grid_values(('uncertainty',), 'posterior')[0]
        elif method == 'prior':
            return self.get_grid_values(('uncertainty',), 'prior')[0]
        elif method == 'predicted':
            return self.get_grid_values(('uncertainty',), 'predicted')[0]

    def get_grid_ratio(self, method: str = 'local'):
        """
        Shortcut for get_grid_values, if you are only interested in ratio.

        Parameters
        ----------
        method
            one of 'local', 'posterior', 'prior', 'predicted'.  See get_grid_values for more info.

        Returns
        -------
        np.ndarray
            2d numpy array of node ratio values
        """

        if method == 'local':
            return self.get_grid_values(('ratio',), 'local')[0]
        elif method == 'posterior':
            return self.get_grid_values(('ratio',), 'posterior')[0]
        elif method == 'prior':
            return self.get_grid_values(('ratio',), 'prior')[0]
        elif method == 'predicted':
            return self.get_grid_values(('ratio',), 'predicted')[0]

    def get_grid_number_hypotheses(self):
        """
        Shortcut for get_grid_values, if you are only interested in the number of hypotheses.  This is much faster than
        using get_grid_values if you only want hypotheses count, as it skips the expensive hypothesis selection logic.

        Returns
        -------
        np.ndarray
            2d numpy array of node hypothesis count
        """

        # you could use self.get_grid_values(['n_hypotheses'], 'local') to get the answer
        #  but this would do the extra logic for determining the best hypothesis that is unnecessary here
        #  let's shortcut this process to make it faster
        data = np.full((self.num_rows, self.num_columns), self.no_data_value)
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                node = self.grid[row][col]
                data[row, col] = node.return_number_of_hypotheses()
        return data

    def get_grid_depth_and_uncertainty(self, method: str = 'local'):
        """
        Shortcut for get_grid_values, if you are only interested in depth and uncertainty.

        Parameters
        ----------
        method
            one of 'local', 'posterior', 'prior', 'predicted'.  See get_grid_values for more info.

        Returns
        -------
        list
            list of 2d numpy array of node depth and uncertainty values, in that order
        """

        if method == 'local':
            return self.get_grid_values(('depth', 'uncertainty',), 'local')
        elif method == 'posterior':
            return self.get_grid_values(('depth', 'uncertainty',), 'posterior')
        elif method == 'prior':
            return self.get_grid_values(('depth', 'uncertainty',), 'prior')
        elif method == 'predicted':
            return self.get_grid_values(('depth', 'uncertainty',), 'predicted')


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

    cp = CubeParameters()
    cp.initialize(iho_order=iho_order, grid_resolution_x=grid_resolution_x, grid_resolution_y=grid_resolution_y)
    for kpam, kval in kwargs.items():
        if kpam in cp.__dir__():
            setattr(cp, kpam, kval)
    if method in ['local', 'posterior', 'prior', 'predicted']:
        cg = CubeGrid(minimum_easting=minimum_easting, maximum_northing=maximum_northing, num_rows=num_rows,
                      num_columns=num_columns, resolution_x=grid_resolution_x, resolution_y=grid_resolution_y,
                      param=cp, use_queue=True, debug=False)
        cg.insert_points(depth, horizontal_uncertainty, vertical_uncertainty, easting, northing)
        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = cg.get_grid_values(('depth', 'uncertainty', 'ratio', 'n_hypotheses'), method=method)
    else:
        raise NotImplementedError(f"run_cube_gridding: {method} not supported, expected one of 'local', 'posterior', 'prior', 'predicted'")
    return depth_grid, uncertainty_grid, ratio_grid, numhyp_grid


if __name__ == '__main__':
    from time import perf_counter
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
    _depth_grid, _uncertainty_grid, _ratio_grid, _numhyp_grid = run_cube_gridding(_z, _thu, _tvu, _x, _y, _numcols,
                                                                                  _numrows,
                                                                                  min(_x), max(_y), 'local', 'order1a',
                                                                                  _resolution_x, _resolution_y)

    endtime = perf_counter()
    print('****CUBE complete: {}****'.format((endtime - starttime)))