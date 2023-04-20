from data import Data


class BaseExplainer:
    def __init__(self):
        self.best = None
        self.rest = None

    def xpln(self, data: Data, best: Data, rest: Data):
        raise NotImplementedError("Cannot create object of BaseExplainer")

    @staticmethod
    def selects(rule, data: Data):
        raise NotImplementedError("Cannot create object of BaseExplainer")
