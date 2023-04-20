from typing import Dict, Union

from data import Data


class BaseExplainer:
    def __init__(self):
        self.best = None
        self.rest = None

    def xpln(self, data: Data, best: Data, rest: Data) -> Union[Dict[str, list], int]:
        raise NotImplementedError("Cannot create object of BaseExplainer")
