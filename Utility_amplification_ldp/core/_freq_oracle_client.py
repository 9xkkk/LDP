
class FreqOracleClient:
    def __init__(self, Epsilon, d, index_mapper=None):
        """

        Args:
            Epsilon: list of privacy budget
            d: int - domain size
            index_mapper: Optional function - maps data items to indexes to in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.Epsilon = Epsilon
        self.d = d

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def update_params(self, Epsilon=None, d=None, index_mapper=None):
        """
        Method to update params of freq oracle client, should be overridden is more options needed
        Args:
            Epsilon: (Optional) dict(list[category data]) - group of privacy budget and data type
            d: (Optional) int - data domain
            index_mapper: (Optional) function - inder_mapper function
        """
        self.Epsilon = Epsilon if Epsilon is not None else self.Epsilon
        self.d = d if d is not None else self.d
        self.index_mapper = index_mapper if index_mapper is not None else self.index_mapper

    def _perturb(self, data):
        """
        Used internally to perturb raw data, must be implemented by a FreqOracle
        Args:
            data: user's data item
        """
        raise NotImplementedError("Must implement")

    def privatise(self, data):
        """
        Public facing method to privatise user's data
        Args:
            data: user's data item
        """
        raise NotImplementedError("Must implement")
