import math
import random
from options import options
from num import Num


def RX(t,s) :
    t = sorted(t)
    return {"name":s or "", 
            "rank":0, 
            "n":len(t), 
            "show":"", 
            "has":t} 

def mid(t):
    t= t.get("has", t)
    n = len(t)//2
    return (t[n] +t[n+1])/2 if len(t)%2==0 else t[n+1]

def div(t):
    t= t.get("has", t)
    return (t[ len(t)*9//10 ] - t[ len(t)*1//10 ])/2.56


def delta(i, other):
    e, y, z = 1E-32, i, other
    return abs(y.mu - z.mu) / ((e + y.sd ** 2 / y.n + z.sd ** 2 / z.n) ** .5)

def merge(rx1,rx2) :
    rx3 = RX([], rx1['name'])
    for _,t in enumerate([rx1['has'],rx2['has']]):
        for _,x in enumerate(t): 
            rx3['has'].append(x)
    rx3['has'] = sorted(rx3['has'])
    rx3['n'] = len(rx3['has'])
    return rx3

def samples(t,n=0):
    u= []
    n = n or len(t)
    # print(n)
    for i in range(n): 
        u.append(t[random.randrange(len(t))]) 
    return u

def gaussian(mu,sd): #  #--> n; return a sample from a Gaussian with mean `mu` and sd `sd`
    mu,sd = mu or 0, sd or 1
    sq,pi,log,cos,r = math.sqrt,math.pi,math.log,math.cos,random.random
    return  mu + sd * sq(-2*log(r())) * cos(2*pi*r())


class ScottKnott:
    def __init__(self, rxs):
        self.rxs = rxs

        self.cohen = None

    def run(self):
        self.rxs = sorted(self.rxs, key=lambda x: mid(x))
        self.cohen = div(self.merges(0, len(self.rxs)-1)) * options["cohen"]

        self.recurse(0, len(self.rxs) - 1, 1)

        return self.rxs
    
    def merges(self, i, j):
        out = RX([], self.rxs[i]['name'])

        for k in range(i, j + 1):
            out = merge(out, self.rxs[j])

        return out

    def same(self, lo, cut, hi):
        l = self.merges(lo, cut)
        r = self.merges(cut + 1, hi)

        return cliffsDelta(l["has"], r["has"]) and bootstrap(l["has"], r["has"])

    def recurse(self, lo, hi, rank):
        cut = None
        b4 = self.merges(lo, hi)
        best = 0

        for j in range(lo, hi + 1):
            if j < hi:
                l = self.merges(lo, j)
                r = self.merges(j + 1, hi)

                now = (l["n"] * (mid(l) - mid(b4)) ** 2 + r["n"] * (mid(r) - mid(b4)) ** 2) / (l["n"] + r["n"])

                if now > best:
                    if abs(mid(l) - mid(r)) >= self.cohen:
                        cut, best = j, now

        if cut and not self.same(lo, cut, hi):
            rank = self.recurse(lo, cut, rank) + 1
            rank = self.recurse(cut + 1, hi, rank)
        else:
            for i in range(lo, hi + 1):
                self.rxs[i]["rank"] = rank

        return rank

def bootstrap(y0, z0):
    x, y, z, yhat, zhat = Num(), Num(), Num(), [], []

    for y1 in y0:
        x.add(y1)
        y.add(y1)

    for z1 in z0:
        x.add(z1)
        z.add(z1)

    xmu, ymu, zmu = x.mu, y.mu, z.mu

    for y1 in y0:
        yhat.append(y1 - ymu + xmu)

    for z1 in z0:
        zhat.append(z1 - zmu + xmu)

    tobs = delta(y, z)
    n = 0

    for _ in range(options["Bootstrap"]):

        if delta(Num(t=samples(yhat)), Num(t=samples(zhat))) > tobs:
            n += 1

    return n / options["Bootstrap"] >= options["Conf"]



def cliffsDelta(ns1, ns2):
    n,gt,lt = 0,0,0
    if len(ns1)> 128 : 
        ns1 = samples(ns1,128) 
    if len(ns2)> 128 : 
        ns2 = samples(ns2,128)
    for x in ns1:
        for y in ns2:
            n = n + 1
            if x > y : 
                gt = gt + 1
            if x < y : 
                lt = lt + 1
    return abs(lt - gt)/n <= options["Cliff"]