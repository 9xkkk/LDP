from approximate_ldp.core import FreqOracleServer
import math

class DEServer(FreqOracleServer):
    def __init__(self, epsilon, deta, d, index_mapper=None):
        """
        Args:
            epsilon: float - the privacy budget
            deta: float [0.0, 1.0] - the privacy relaxing degree
            d: integer - the size of the data domain
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, deta, d, index_mapper=index_mapper)
        self.update_params(epsilon, deta, d, index_mapper)

    def update_params(self, epsilon=None, deta=None, d=None, index_mapper=None):
        """
        Updates DEServer parameters. This will reset any aggregated/estimated data
        Args:
            epsilon: optional - privacy budget
            deta: optional - privacy relaxing degree
            d: optional - domain size
            index_mapper: optional - function

        """
        super().update_params(epsilon, deta, d, index_mapper)
        if epsilon is not None or deta is not None or d is not None:
            self.const = math.pow(math.e, self.epsilon) + self.d - 1
            self.p = (math.pow(math.e, self.epsilon) + self.deta * (self.d - 1)) / self.const
            self.q = (1 - self.deta) / self.const

    def aggregate(self, priv_data):
        """
        Used to aggregate privatised data by DEClient.privatise
        Args:
            priv_data:  privatised data from DEClient.privatise
        """
        self.aggregated_data[priv_data] += 1
        self.n += 1

    def _update_estimates(self):
        self.estimated_data = (self.aggregated_data - self.n * self.q) / (self.p - self.q)
        return self.estimated_data

    def estimate(self, data, suppress_warnings=False):
        """
        Calculate a frequency estimate of the given data item
        Args:
            data: data item
            suppress_warnings: Optional boolean - Suppresses warning about possible inaccurate estimations

        Returns: float - frequency estimate

        """
        self.check_warning(suppress_warnings)
        index = self.index_mapper(data)
        self.check_and_update_estimates()
        return self.estimated_data[index]