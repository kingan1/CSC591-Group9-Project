from explain import Explain, show_rule, selects
from data import Data
from options import options
from discretization import bins, value
from num import Num
from sym import Sym
from utils import adds, set_seed, rint, rand, rnd, csv, cliffsDelta, showTree, diffs

help = """

xpln: multi-goal semi-supervised explanation
(c) Group 9
  
USAGE: python3 main.py [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliffs  cliff's delta threshold      = .147
  -d  --d       different is over sd*d       = .35
  -f  --file    data file                    = ../data/auto93.csv
  -F  --Far     distance to distant          = .95
  -g  --go      start-up action              = nothing
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 4
  -R  --Reuse   child splits reuse a parent pole = true
  -s  --seed    random number seed           = 937162211
"""


def main(funs, saved=None, fails=None):
    """
    `main` fills in the settings, updates them from the command line, runs
    the start up actions (and before each run, it resets the random number seed and settongs);
    and, finally, returns the number of test crashed to the operating system.

    :param funs: list of actions to run
    :param saved: dictionary to store options
    :param fails: number of failed functions
    """

    saved, fails = {}, 0
    options.parse_cli_settings(help)

    for k, v in options.items():
        saved[k] = v

    if options['help']:
        print(help)
    else:
        for what, fun in funs.items():
            if options['go'] == "all" or what == options['go']:
                for k, v in saved.items():
                    options[k] = v

                if funs[what]() is False:
                    fails = fails + 1
                    print("❌ fail:", what, "-"*60)
                else:
                    print("✅ pass:", what, "-"*60)
    exit(fails)


# Examples
egs = {}


def eg(key, s, fun):
    global help
    egs[key] = fun
    help += "  -g  {}\t{}\n".format(key, s)


def check_the():
    return str(options)


def check_rand():
    set_seed(1)
    t = []

    for i in range(1, 1000 + 1):
        t.append(rint(100))

    set_seed(1)
    u = []

    for i in range(1, 1000 + 1):
        u.append(rint(100))

    for k, v in enumerate(t):
        assert v == u[k]


def check_some():
    options['Max'] = 32
    num1 = Num()
    set_seed(options['seed'])
    for i in range(1, 10000 + 1):
        num1.add(i)

    print(num1.has())


def check_nums():
    num1, num2 = Num(), Num()

    set_seed(options["seed"])
    for i in range(1, 10000 + 1):
        num1.add(rand())

    for i in range(1, 10000 + 1):
        num2.add(rand() ** 2)

    print(1, rnd(num1.mid()), rnd(num1.div()))
    print(2, rnd(num2.mid()), rnd(num2.div()))

    return .5 == rnd(num1.mid()) and num1.mid() > num2.mid()


def check_syms():
    sym = Sym()
    adds(sym, ["a", "a", "a", "a", "b", "b", "c"])

    print(sym.mid(), rnd(sym.div()))

    return 1.38 == rnd(sym.div())


def check_csv():
    n = 0

    def f(t):
        nonlocal n
        n += len(t)

    csv(options['file'], f)
    return 3192 == n


def check_data():
    data = Data()
    data.read(options['file'])

    col = data.cols.x[0]

    print(col.lo, col.hi, col.mid(), col.div())
    print(data.stats())


def check_clone():
    data1 = Data()
    data1.read(options['file'])

    data2 = data1.clone(data1, data1.rows)

    print(data1.stats())
    print(data2.stats())


def check_cliffs():
    assert not cliffsDelta([8, 7, 6, 2, 5, 8, 7, 3], [8, 7, 6, 2, 5, 8, 7, 3]), "1"
    assert cliffsDelta([8, 7, 6, 2, 5, 8, 7, 3], [9, 9, 7, 8, 10, 9, 6]), "2"

    t1, t2 = [], []

    for i in range(1, 1000 + 1):
        t1.append(rand())

    for i in range(1, 1000 + 1):
        t2.append(rand() ** .5)

    assert not cliffsDelta(t1, t1), "3"
    assert cliffsDelta(t1, t2), "4"

    diff, j = False, 1.0

    while not diff:
        def f(x):
            nonlocal j
            return x * j

        t3 = map(f, t1)
        diff = cliffsDelta(t1, list(t3))
        print(">", rnd(j), diff)
        j = j * 1.025


def check_dist():
    data = Data()
    data.read(options['file'])

    num = Num()

    for _, row in enumerate(data.rows):
        num.add(data.dist(row, data.rows[0]))

    d = {"lo": num.lo, "hi": num.hi, "mid": rnd(num.mid()), "div": rnd(num.div())}
    print(d)


def check_half():
    data = Data()
    data.read(options['file'])
    set_seed(options['seed'])

    left, right, A, B, c, _ = data.half()
    print(len(left), len(right))

    l, r = Data.clone(data, left), Data.clone(data, right)
    print("l", l.stats())
    print("r", r.stats())


def check_tree():
    data1 = Data()
    data1.read(options['file'])
    set_seed(options['seed'])

    showTree(data1.tree())


def check_sway():
    data = Data()
    data.read(options['file'])
    set_seed(options['seed'])

    best, rest, _ = data.sway()

    print("\nall ", data.stats())
    print("    ", data.stats(what="div"))
    print("\nbest", best.stats())
    print("    ", best.stats(what="div"))
    print("\nrest", rest.stats())
    print("    ", rest.stats(what="div"))
    print("\nall ~= best?", diffs(best.cols.y, data.cols.y, options))
    print("best ~= rest?", diffs(best.cols.y, rest.cols.y, options))


def check_bins():
    data = Data()
    data.read(options['file'])
    set_seed(options['seed'])

    best, rest,_ = data.sway()
    print("all", "", "", "", {"best": len(best.rows), "rest": len(rest.rows)})

    b4 = None
    for k, t in enumerate(bins(data.cols.x, {"best": best.rows, "rest": rest.rows})):
        for _, range_ in enumerate(t):
            if range_.txt != b4:
                print()

            b4 = range_.txt

            print(
                range_.txt, range_.lo, range_.hi,
                rnd(value(range_.y.has, n_b=len(best.rows), n_r=len(rest.rows), s_goal="best")),
                dict(range_.y.has)
            )

def check_xpln():
    data=Data()
    data.read(options['file'])
    set_seed(options['seed'])
    best,rest,evals = data.sway()
    x = Explain(best, rest)
    rule,most= x.xpln(data,best,rest)
    print("\n-----------\nexplain=", show_rule(rule))
    data1= Data()
    data1.read(data,selects(rule,data.rows))
    print("all               ",data.stats(),data.stats(what="div"))
    print("sway with {:5} evals".format(evals),(best.stats()),(best.stats(what="div")))
    print("xpln on {:5} evals".format(evals),(data1.stats()),(data1.stats(what="div")))
    top2,_ = data.betters(len(best.rows))
    top = Data()
    top.read(data,top2)
    print("sort with {:5} evals".format(len(data.rows)) ,(top.stats()), (top.stats(what="div")))


eg("the", "show options", check_the)
eg("rand", "demo random number generation", check_rand)
eg("some", "demo of reservoir sampling", check_some)
eg("nums", "demo of Num", check_nums)
eg("syms", "demo SYMS", check_syms)
eg("csv", "reading csv files", check_csv)
eg("data", "showing data sets", check_data)
eg("clone", "replicate structure of a Data", check_clone)
eg("cliffs", "stats tests", check_cliffs)
eg("dist", "distance test", check_dist)
eg("half", "divide data in halg", check_half)
eg("tree", "make snd show tree of clusters", check_tree)
eg("sway", "optimizing", check_sway)
eg("bins", "find deltas between best and rest", check_bins)
eg("xpln","explore explanation sets", check_xpln)

main(egs)
