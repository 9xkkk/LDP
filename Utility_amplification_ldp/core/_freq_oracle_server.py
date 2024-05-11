import warnings
import numpy as np

class FreqOracleServer:
    def __init__(self, Epsilon, d, inder_mapper=None):
        """

        Args:
            Epsilon: list of privacy budget
            d: int - data domain
            inder_mapper: (Optional) function - index_mapper function
        """
        self.Epsilon = Epsilon
        self.d = d

        self.aggregated_data = np.zeros(self.d)  # Some freq oracle servers keep track of aggregated data to generate estimated_data
        self.estimated_data = np.zeros(self.d)  # Keep track of estimate data for quick access
        self.n = 0

        self.name = "FrequencyOracle"  # Name of the frequency oracle for warning messages, set using .set_name(name)
        self.last_estimated = 0

        if inder_mapper is not None:
            self.index_mapper = inder_mapper
        else:
            self.index_mapper = lambda x: x-1

    def set_name(self, name):
        """
        Sets freq servers name
        Args:
            name: string - name of frequency oracle

        """
        self.name = name

    def reset(self):
        """
        This method resets the server's aggregated/estimated data and sets n =0
        This should be overridden if other parameters need to be reset.
        """
        self.aggregated_data = np.zeros(self.d)
        self.estimated_data = np.zeros(self.d)
        self.last_estimated = 0
        self.n = 0

    def update_params(self, Epsilon=None, d=None, index_mapper=None):
        """
        Method to update parameters of freq oracle server, should be overridden if more options needed.
        Args:
            Epsilon: Optional - privacy budget and data type
            d: Optional - domain size
            index_mapper: Optional - function
        """
        self.Epsilon = Epsilon if Epsilon is not None else self.Epsilon # Updating Epsilon here will not update any internal probabilities
        # Any class that implements FreqOracleServer, need to override update_params to update Epsilon properly

        self.d = d if d is not None else self.d
        self.index_mapper = index_mapper if index_mapper is not None else self.index_mapper
        self.reset()

    def check_warnings(self, suppress_warnings=False):
        """
        Used during estimation to check warnings
        Args:
            suppress_warnings: Optional boolean - If True suppresses warnings from being output
        """
        if not suppress_warnings:
            if self.n < 10000:
                warnings.warn(self.name + "has only aggregated small amounts of data (n=" + str(self.n) +
                              ") estimations may be highly inaccurate on small datasets", RuntimeWarning)

    def aggregate(self, data):
        """
        The main method for aggregation, should be implemented by a freq oracle server
        Args:
            data: item to estimate frequency of
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

    def check_and_uopdate_estimates(self):
        """
        Used to check if the "cached" estimated data needs re-estimating, this occurs when new data has been aggregated since last
        """
        if self.last_estimated < self.n: # If new data has been aggregated since the last estimation then estimate all
            self.last_estimated = self.n
            self._update_estimates()

    def _update_estimates(self):
        """
        Used internally to update estimates, should be implemented
        """
        raise NotImplementedError("Must implement")

    def estimate(self, data, suppress_warnings=False):
        """
        Calculate frequency estimate of given data item, must be implemented
        Args:
            data: data to estimate the frequency warning of
            suppress_warnings: Optional boolean - if true suppresses warnings
        """
        raise NotImplementedError("Must implement")

    @property
    def get_estimates(self):
        """
        Returns: Estimate data
        """
        return self.estimated_data
