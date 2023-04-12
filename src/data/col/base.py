class Col:
    def __init__(self, at: int = 0, txt: str = ""):
        self.at = at
        self.txt = txt

        self.n = 0

    def add(self, x, n: int = 1):
        raise NotImplementedError("Cannot create object of Col")

    def mid(self):
        raise NotImplementedError("Cannot create object of Col")

    def div(self):
        raise NotImplementedError("Cannot create object of Col")

    def dist(self, data1, data2):
        raise NotImplementedError("Cannot create object of Col")
