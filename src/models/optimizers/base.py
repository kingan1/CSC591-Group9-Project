from data import Data
from utils import set_seed, Random, get_seed


class BaseOptimizer:
    def __init__(self, seed: int = None):
        self._random = Random()

        if seed:
            self._random.set_seed(seed)

    def run(self, data: Data):
        set_seed(self._random.seed)
        best, rest, evals = self._run(data=data)
        self._random.set_seed(get_seed())

        return best, rest, evals

    def _run(self, data: Data):
        raise NotImplementedError("Cannot create an object of BaseOptimizer")