import numpy as np
import math
import random

from Utility_amplification_ldp.core import FreqOracleClient

# Client-side for MIDUE-encoding
    # By default parameters are set for opt0: Optimization Model in the Worst-Case
    # If which_opt=1 is passed to the constructor then it uses opt1: Optimization Model Constrained with RAPPOR Structure.
    # If which_opt=1 is passed to the constructor then it uses opt2: Optimization Model Constrained with OUE Structure.

class MIDUEClient(FreqOracleClient):
    def __init__(self, Epsilon, d, which_opt=0, index_mappr=None):
        """

        Args:
            Epsilon: dict(list[category]) - group of privacy budget and data type
            d: integer - the size of data domain
            which_opt: Optional boolean which is 0, 1 or 2
            index_mappr: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(Epsilon, d, index_mapper=index_mappr)
        self.which_opt = which_opt
        self.update_params(Epsilon, d, index_mapper=index_mappr)

    def update_params(self, Epsilon=None, d=None, index_mapper=None):
        """
        Used to update the client MIDUE parameters.
        Args:
            Epsilon: optional - group of the privacy budget and data type
            d: optional - the size of data domain
            index_mapper: optional - function
        """
        super().update_params(Epsilon, d, index_mapper)

        if Epsilon is not None:
            self.probabilities = self._Calculate_optimized_probabilities()
            self.p = self.probabilities[0]
            self.q = self.probabilities[1]

    def _Calculate_optimized_probabilities(self):

        if self.which_opt == 0:
            pass
        elif self.which_opt == 1:
            pass
        else:
            pass


    def _perturb(self, index):
        """
        Used internally to perturb data using MIDUE-ldp
        Args:
            index: the index corresponding to the data item

        Returns: privatised data vector

        """
        oh_vec = []
        for i in range(len(self.d)):
            oh_bit = np.random.choice([1, 0], size=1, p=[self.q[i], self.p[i]])
            oh_vec.append(oh_bit)
        oh_vec[index] = 0
        if random.random() < self.p[index]:
            oh_vec[index] = 1
        return oh_vec

    def privatise(self, data):
        """
        Privatises a user's data item using MIDUE encoding.
        Args:
            data: data item

        Returns: privatised data vector

        """
        index = self.index_mapper(data)
        return self._perturb(index)