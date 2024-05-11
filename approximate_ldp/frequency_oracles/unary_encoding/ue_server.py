import math
import numpy as np

from approximate_ldp.core._freq_oracle_server import FreqOracleServer

class UEServer(FreqOracleServer):
    def __init__(self, epsilon, deta, d, use_oue=False, index_mapper=None):
        """
        Args:
            epsilon: float - privacy budget
            deta: float [0.0, 1.0] - privacy relaxing degree
            d: integer - domain size
            use_oue: Optional boolean - If True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, deta, d, index_mapper=index_mapper)
        self.set_name("UEServer")
        self.use_oue = use_oue
        self.update_params(epsilon, deta, d, index_mapper)

    def update_params(self, epsilon=None, deta=None, d=None, index_mapper=None):
        """
        Updates UE server parameters. This will reset any aggregated/estimated data
        Args:
            epsilon:  optional - privacy budget
            deta: optional - privacy relaxing degree
            d: optional - domain size
            index_mapper: optional - index_mapper function
        """
        super().update_params(epsilon, deta, d, index_mapper)

        if epsilon is not None or deta is not None:
            self.const = (math.pow(math.e, self.epsilon) + self.deta) ** 0.5 + 1
            self.p = (math.pow(math.e, self.epsilon) + self.deta) ** 0.5 / self.const
            self.q = 1 / self.const

            if self.use_oue is True:
                self.p = 0.5
                self.q = 1 / (math.pow(math.e, self.epsilon) + self.deta + 1)

    def aggregate(self, priv_data):
        """
        Used to aggregate privatised data by ue_client.privatise
        Args:
            priv_data: privatised data from ue_client.privatise
        """
        self.aggregated_data += priv_data
        self.n += 1

    def _update_estimates(self):
        self.estimated_data = (self.aggregated_data - self.n * self.q) / (self.p - self.q)
        return self.estimated_data

    def estimate(self, data, suppress_warnings=False):
        """
        Calculate a frequency estimate of the given data item
        Args:
            data: data item
            suppress_warnings: Optional boolean - Supresses warning about possible inaccurate estimation

        Returns: float - frequency estimate

        """
        self.check_warning(suppress_warnings)
        index = self.index_mapper(data)
        self.check_and_update_estimates()
        return self.estimated_data[index]
