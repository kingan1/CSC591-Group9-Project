from data import Data


class BaseOptimizer:
    def run(self, data: Data):
        raise NotImplementedError("Cannot create an object of BaseOptimizer")
