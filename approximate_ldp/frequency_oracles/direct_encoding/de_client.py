from approximate_ldp.core import FreqOracleClient
import math
import numpy as np
import random


class DEClient(FreqOracleClient):
    def __init__(self, epsilon, deta, d, index_mapper=None):
        super().__init__(epsilon, deta, d, index_mapper)
        self.update_params(epsilon, deta, d, index_mapper)

    def update_params(self, epsilon=None, deta=None, d=None, index_mapper=None):
        """
        Used to update the client DE parameters
        Args:
            epsilon: optional - privacy budget
            deta: optional - privacy relaxing degree
            d: optional - domain size
            index_mapper: optional - function
        """
        super().update_params(epsilon, deta, d, index_mapper)

        if epsilon is not None or deta is not None or d is not None: # If epsilon or deta changes, update probs
            self.const = math.pow(math.e, self.epsilon) + self.d - 1
            self.p = (math.pow(math.e, self.epsilon) + self.deta * (self.d - 1)) / self.const
            self.q = (1 - self.deta) / self.const

    def _perturb(self, data):
        if random.random() < self.p:
            return data
        else:
            perturbed_data = random.randint(0, self.d - 2)
            if perturbed_data == data:
                return self.d - 1
            else:
                return perturbed_data

    def privatise(self, data):
        """
        Privatises a user'data item using Direct Encoding (DE)
        Args:
            data: data item

        Returns: privatised data vector

        """
        index = self.index_mapper(data) # Maps data to the range {0, ..., d-1}
        return self._perturb(index)

