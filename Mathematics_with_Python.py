"""
We can do a whole bunch of really cool math with python but first we need to
import some modules so we can use the cool math functions that are
predefined in python
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import scipy as sp
from scipy import *
"""
These modules allow us to do things like...
"""
p = np.pi
print(p)

a = np.sqrt((3*37**5)/356+29)
print(a)
"""
So with numpy we can do some interesting math... like a basic caluclator but
faster and better.... We can also do things like
"""

y = np.linspace(-10,10,100)
c = [ x for x in range(100)]
b = np.sum(c)

print(b)

f = np.diff(c)
print(f)
"""
Computing sums and differences with just two lines!!
also we can do things like hyperbolic trigometry and stuff!
"""
g = np.sinh(y)
h = np.cosh(y)

"""
Another really cool thing is given a list, we can take the derivative and
integral of a list and plot the results!!
"""

import itertools as it
a = list(i for i in range(100))
d = list(it.accumulate(a))

e = np.gradient(d)

plt.figure()
plt.plot(a,d)
plt.show()

plt.figure()
plt.plot(a,e)
plt.show()

"""
As you can see, python can double as a caluclator, calculus sovler and graphing
program all as you see fit. We can also do things like make spread sheets by using
pythons pandas package that allows us to handle very large data sets. Like if you
have a data set on several hard drives, then calling the python pandas package to
do things like compute averages, standard deviations, tabulate the data, and plot
it, and it will do this with ease and effiency.
"""

import pandas as pd
from pandas import DataFrame
names = list(i for i in range(100))
#["Bob", "Bill", "Alex", "Joe", "Roger", "Tom"]
TestScores = list(x for x in range(10))

Data = list(zip(names,TestScores))

print(Data[0:1])
d = {"Dan":[1,2],"Pat":[3,4],"Bob":[5,6]}
data = pd.DataFrame(data=Data)
print(data)

data.mean(0)
data.std(0)
plt.figure; data.plot.bar(style='k') ;plt.show()

"""
as you can see, we can do some cool math and statisics with python pandas.
We can make dataframes for almost anything. They are like numpy data arrays
but better cause the build just like a spreadsheet and they look really nice
and clean. Pandas was accutally built off of numpy but the developers added a
whole bunch of new tools and fixed a lot of the bugs within numpy. But that doesn't
mean numpy is totally useless, we can do discrete Fourier Transforms or DFT, and
Fast Fourier Transforms or FFTs using numpy.
"""

def F(j):
    return 2*np.sin(j)**2 + 3*np.cos(3*j/2)+ 1.5* np.sin(9*j)
x = np.linspace(0,2*np.pi,100)
F = F(x)
plt.figure
plt.plot(x,F)
plt.show()

fft = np.real(np.fft.fft(F,n=None))
plt.figure
plt.plot(x,fft)
plt.show()
"""
So as you can see, numpy is not toatlly useless. In fact numpy and scipy packages
make a very good linear algebra package. Some of the basic things we can do with
the scipy linalg package is dot products, cross products, and eigenvalue problems.
"""
import scipy.linalg as spl
from scipy.linalg import *

a = np.array([1,2,3])
print(a)
aT = np.transpose(a)
print(aT)
b = np.array([4,5,6])
c = np.vdot(a,b)
print(c)
"""
So as you can see we can do stuff with vectors quickly and easily. We can also do
lots of fun things with eigenvale and eigenvector problems
"""
d = np.matrix([[1,0,0],[0,2,0],[0,0,3]])
print(d)
f = spl.eig(d)
print(f)

"""
We can also do things like take determinants of a matrix and find thier inverses
"""
g = spl.det(d)
print(g)

i = spl.inv(d)
print(i)

j=i*d
print(j)
"""
All in all we can do a whole bunch of math and physics with the packages python
has to offer. With Pandas we can do things like statistical analysis of data sets.
With Matplotlib and pyplot we can visulize the data and plot functions.
with numpy and scipy we can do a whole bunch of amazing numerical routines that
are useful in all fields of physics.
"""
