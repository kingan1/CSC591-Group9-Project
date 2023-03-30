from typing import List


class Row:
    """
    Stores a row.
    """

    def __init__(self, t: List):
        self.cells = t
        self.x = None
        self.y = None
