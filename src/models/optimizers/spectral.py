from typing import Optional

from data import Data


class SpectralClusteringOptimizer:
    def __init__(self):
        self._data: Optional[Data] = None

    def run(self, data: Data):
        self._data: Data = data
