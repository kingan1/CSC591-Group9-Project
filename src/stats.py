import math
import random
from options import options
from num import Num
from utils import many

random.seed(0)

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
    for i in range(n): 
        u.append(t[random.randrange(len(t))]) 
    return u

def gaussian(mu,sd): #  #--> n; return a sample from a Gaussian with mean `mu` and sd `sd`
    mu,sd = mu or 0, sd or 1
    sq,pi,log,cos,r = math.sqrt,math.pi,math.log,math.cos,random.random
    return  mu + sd * sq(-2*log(r())) * cos(2*pi*r())



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
    if len(ns1) > 256:
        ns1 = many(ns1, 256)
    if len(ns2) > 256:
        ns2 = many(ns2, 256)
    if len(ns1) > 10 * len(ns2):
        ns2 = many(ns1, 10 * len(ns2))
    if len(ns2) > 10 * len(ns1):
        ns2 = many(ns2, 10 * len(ns1))

    n, gt, lt = 0, 0, 0
    for x in ns1:
        for y in ns2:
            n = n + 1
            if x > y:
                gt = gt + 1

            elif x < y:
                lt = lt + 1

    return abs(lt - gt) / n <= options["cliff"]

