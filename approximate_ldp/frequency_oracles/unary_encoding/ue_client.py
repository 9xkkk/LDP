import random
import numpy as np
import math
from approximate_ldp.core._freq_oracle_client import FreqOracleClient

class UEClient(FreqOracleClient):
    def __init__(self, epsilon, deta, d, use_oue=False, index_mapper=None):
        """
        Args:
            epsilon: float - privacy budget
            deta: float [0.0, 1.0] - privacy relaxing degree
            d: integer - the size of the data domain
            use_oue: Optional boolean - if True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, deta, d, index_mapper=index_mapper)
        self.use_oue = use_oue
        self.update_params(epsilon, deta, d, index_mapper)

    def update_params(self, epsilon=None, deta=None, d=None, index_mapper=None):
        """
        Used to update the client UE parameters
        Args:
            epsilon: optional - privacy budget
            deta: optional - privacy relaxing degree
            d: optional - domain size
            index_mapper: optional - function
        """
        super().update_params(epsilon, deta, d, index_mapper)

        if epsilon is not None or deta is not None:
            self.const = (math.pow(math.e, self.epsilon) + self.deta) ** 0.5 + 1
            self.p = ((math.pow(math.e, self.epsilon) + self.deta) ** 0.5) / self.const
            self.q = 1 / self.const

            if self.use_oue is True:
                self.p = 0.5
                self.q = 1 / (math.pow(math.e, self.epsilon) + self.deta + 1)

    def _perturb(self, index):
        """
        Used internally to perturb data using unary encoding
        Args:
            index: the index corresponding to the data item

        Returns: privatised data vector
        """
        oh_vec = np.random.choice([1, 0], size=self.d, p=[self.q, 1-self.q])
        oh_vec[index] = 0
        if random.random() < self.p:
            oh_vec[index] = 1
        return oh_vec

    def privatise(self, data):
        """
        Privatises a user's data item using unary encoding
        Args:
            data: data item

        Returns: privatised data vector

        """
        index = self.index_mapper(data)
        return self._perturb(index)
