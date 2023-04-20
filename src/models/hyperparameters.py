from distance import PDist


class Hyperparameter:
    DEFAULT = {
        "reuse": True,
        "far": 0.95,
        "halves": 512,
        "rest": 10,
        "i_min": 0.5,
        "distance_class": PDist(p=2)
    }

    OPTIMIZED = {
        "reuse": True,
        "far": 0.75,
        "halves": 500,
        "rest": 3,
        "i_min": 0.2,
        "distance_class": PDist(p=2)
    }
