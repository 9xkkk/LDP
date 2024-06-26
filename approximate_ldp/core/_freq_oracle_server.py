import warnings
import numpy as np
from approximate_ldp.core.prob_simplex import project_probability_simplex

class FreqOracleServer:
    def __init__(self, epsilon, deta, d, index_mapper=None):
        """
        Args:
            epsilon: Privacy Budget
            deta: Privacy Relaxing Degree
            d: Domain size - not all freq oracle need this so can be None
            index_mapper: Optional function - maps data item to indexes in the range {0, 1, ..., d-1} where d is the size of data domain
        """
        self.epsilon = epsilon
        self.deta = deta
        self.d = d

        self.aggregated_data = np.zeros(self.d) # Some freq oracle servers keep track of aggregated data to generate estimated_data
        self.estimated_data = np.zeros(self.d) # Keep track of estimated data for quick access
        self.n = 0 # The number of data items aggregated

        self.name = "FrequencyOracle" # Name of the frequency oracle for warning messages, set using .set_name(name)
        self.last_estimated = 0

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def set_name(self, name):
        """
        Sets freq servers name
        Args:
            name: string - name of frequency oracle
        """
        self.name =name

    def reset(self):
        """
        This method resets the server's aggregated/estimated data and sets n = 0.
        This should be overridden if other parameters need to be reset.
        """
        self.aggregated_data = np.zeros(self.d)
        self.estimated_data = np.zeros(self.d)
        self.last_estimated = 0
        self.n = 0

    def update_params(self, epsilon=None, deta=None, d=None, index_mapper=None):
        """
        Method to update params of freq oracle server, should be overridden if more options needed.
        This will reset aggregated/estimated data.
        Args:
            epsilon: Optional - privacy budget
            d: Optional - domain size
            index_mapper: Optional - function
        """
        self.epsilon = epsilon if epsilon is not None else self.epsilon # Update epsilon here will not update any internal probabilities
        # Any class that implements FreqOracleServer, needs to overide update_params to update epsilon properly

        self.deta = deta if deta is not None else self.deta
        self.d = d if d is not None else self.d
        self.index_mapper = index_mapper if index_mapper is not None else self.index_mapper
        self.reset()

    def check_warning(self, suppress_warning=False):
        """
        Used during estimation to check warnings
        Args:
            suppress_warning: Optional boolean - If True suppresses warnings from being output
        """
        if not suppress_warning:
            if self.n < 10000:
                # TODO: This seems to warn too many times in HH + FLH
                warnings.warn(self.name + " has only aggregated small amounts of data (n=" + str(self.n) +
                              ") estimations may be highly inaccurate on small datasets", RuntimeWarning)
            if self.epsilon < 1:
                warnings.warn("High privacy has been detected (epsilon = " + str(self.epsilon) +
                              "), estimations may be highly inaccurate on small datasets", RuntimeWarning)

    def aggregate(self, data):
        """
        The main method for aggregation, should be implemented by a freq oracle server
        Args:
            data:  item to estimate frequency
        """
        raise NotImplementedError("Must implement")

    def aggregate_all(self, data_list):
        """
        Helper method used to aggregate a list of data
        Args:
            data_list: List of private data to aggregate
        """
        for data in data_list:
            self.aggregate(data)

    def check_and_update_estimates(self):


        """
        Used to check if the "cached" estimated data needs re-estimating, this occurs when new data has been aggregated since last
        """
        if self.last_estimated < self.n:  # If new data has been aggregated since the last estimation, then estimate all
            self.last_estimated = self.n
            self._update_estimates()

    def _update_estimates(self):
        """
        Used internally to update estimates,should be implemented
        """
        raise NotImplementedError("Muse implement")

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates frequency estimate of given data item, must be implemented
        Args:
            data:  data to estimate the frequency warning of
            suppress_warnings: Optional boolean - if true suppresses warnings
        """
        raise NotImplementedError("Must implement")

    def estimate_all(self, data_list, suppress_warnings=False, normalization=0):
        """
        Helper method, given a list of data items, return a list of their estimated frequencies
        Args:
            data_list: List of data items to estimate
            suppress_warnings: If True, will suppress estimation warnings
            normalization: Normalization should only be specified when estimating over the entire domain!
                            0 - No Norm
                            1 - Additive Norm
                            2 - Prob Simplex
                            3 (or otherwise) - Threshold cut
        Returns: list of estimates

        """
        self.check_and_update_estimates()

        estimates = np.array([self.estimate(x, suppress_warnings=suppress_warnings) for x in data_list])

        if normalization == 0: # No normalization
            return estimates
        elif normalization == 1: # Additive normalization
            diff = self.n - sum(estimates[estimates > 0])
            non_zero = (estimates > 0).sum()

            for i,item in enumerate(estimates):
                if item >0:
                    estimates[i] = item + diff/non_zero
                else:
                    estimates[i] = 0

            return estimates
        elif normalization == 2: # Prob Simplex
            proj = project_probability_simplex(estimates/self.n)
            return np.array(proj) * self.n
        else:
            # Threshould cut
            sorted_index = np.argsort((-1 * estimates))
            total = 0
            i = 0
            for i,index in enumerate(sorted_index):
                total += estimates[index]
                if total > self.n:
                    break

            for j in range(i, len(sorted_index)):
                estimates[sorted_index[j]] = 0

            return estimates
    @property
    def get_estimate(self):
        """
        Returns: Estimated data

        """
        return self.estimated_data
