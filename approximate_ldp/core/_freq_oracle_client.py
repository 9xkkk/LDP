
class FreqOracleClient:
    def __init__(self, epsilon, deta, d, index_mapper=None):
        """

        Args:
            epsilon (float): Privacy budget
            deta (float): Privacy Relaxing Degree
            d (int): domain size - not all freq oracles need this, so can be None
            index_mapper (func): Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.epsilon = epsilon
        self.deta = deta
        self.d = d

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1
        else:
            self.index_mapper = index_mapper

    def update_params(self, epsilon=None, deta=None, d=None, index_mapper=None):
        """

        Method to updata params of freq oracle client, should be overridden if more options needed
        Args:
            epsilon (optional float): Privacy budget
            deta (optional float): Privacy relaxing degree
            d (optional int): Domain size
            index_mapper (optional func): Index map function
        Returns:
        """
        self.epsilon = epsilon if epsilon is not None else self.epsilon
        self.deta = deta if deta is not None else self.deta
        self.d = d if d is not None else self.d

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

