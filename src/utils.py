import copy as cp
import io
import math
import re
import sys



class Random:
    def __init__(self):
        self.seed = 937162211

    def set_seed(self, value: int):
        self.seed = value

    def rand(self, lo=0, hi=1):
        """
        Generates a pseudo-random number using seed.

        :param lo: Lower limit of generated number
        :param hi: Higher limit of generated number
        :return: Pseudo-random number
        """

        self.seed = (16807 * self.seed) % 2147483647
        return lo + (hi - lo) * self.seed / 2147483647

    def rint(self, lo=0, hi=1):
        return math.floor(0.5 + rand(lo, hi))


_inst = Random()
rand = _inst.rand
rint = _inst.rint
set_seed = _inst.set_seed


def rnd(n: float, n_places: int = 2) -> float:
    """
    Rounds number n to n places.

    :param n: Number
    :param n_places: Number of decimal places to round
    :return: Rounded number
    """
    mult = math.pow(10, n_places)
    return math.floor(n * mult + 0.5) / mult


def coerce(v):
    """
    Attempts to convert v to an int, float, bool, or keep as string

    :param v: String to convert
    :return: v converted to its type
    """
    types = [int, float]

    for t in types:
        try:
            return t(v)
        except ValueError:
            pass

    bool_vals = ["true", "false"]
    if v.lower() in bool_vals:
        return v.lower() == "true"

    return v


def csv(sFilename, fun):
    """
    call `fun` on rows (after coercing cell text)

    :param sFilename: String of the file to read
    :param fun: function to call per each row
    """
    f = io.open(sFilename)
    while True:
        s = f.readline().rstrip()
        if s:
            t = []
            for s1 in re.findall("([^,]+)", s):
                t.append(coerce(s1))
            fun(t)
        else:
            return f.close()


def many(t, n):
    """
    returns some items from `t`
    """
    return [any(t) for _ in range(n)]


def any(t):
    """
    returns one items at random
    """
    return t[rint(len(t)) - 1]


def copy(t):
    return cp.deepcopy(t)


def norm(num, n):
    return n if n == "?" else (n - num.lo) / (num.hi - num.lo + sys.float_info.min)


def per(t, p):
    p = math.floor(((p or 0.5) * len(t)) + 0.5)
    return t[max(0, min(len(t), p) - 1)]


def kap(t, fun, u={}):
    u = {}
    what = enumerate(t)
    if type(t) == dict:
        what = t.items()
    for k, v in what:
        v, k = fun(k, v)
        if not k:
            u[len(u)] = v
        else:
            u[k] = v
    return u


