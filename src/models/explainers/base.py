from data import Data
from utils import timed


class BaseExplainer:
    def __init__(self):
        self.best = None
        self.rest = None

    def _xpln(self, data: Data, best: Data, rest: Data):
        raise NotImplementedError("Cannot create object of BaseExplainer")

    @timed
    def xpln(self, *args, **kwargs):
        return self._xpln(*args, **kwargs)

    @staticmethod
    def selects(rule, data: Data):
        raise NotImplementedError("Cannot create object of BaseExplainer")
