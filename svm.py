import numpy, random ,math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import namedtuple

global t
global P
global N
global C

def zerofun(alfa):
    " Defines the upper bound to incorporate slack variables "
    fc = all(0 <= a and a <= C for a in alfa)
    sc = sum(numpy.dot(alfa, t)) == 0
    return fc and sc

def initialize_P(x, t, K):
    "Initialize the P matrix used in objective func"
    global P
    P = numpy.array([[t[i]*t[j]*K(x[i], x[j]) for i in range(0,N)] for j in range(0,N)])

def objective(alfa):
    return numpy.dot(alfa, P)/2 - numpy.sum(alfa)

def K(a,b):
    return linear_kernel(a,b)

def linear_kernel(a, b):
    return numpy.dot(a,b)

# A data point along with its corresponding target and alpha value
DataPointInfo = namedtuple('DataPointInfo', ['point','target','alpha'])

def non_zero(n):
    "True if the number is separate from 0 with a small epsilon"
    return abs(n) > 0.00001

def is_support_vector(p):
    "True iff the point in the DataPointInfo object is a support vector"
    return non_zero(p.alpha)

def support_vectors(ps):
    "Filter only the support vectors from the data points"
    return filter(is_support_vector, ps)

def ind_no_b(s, ps, K):
    """The summation that appears in the indicator function,
    as well as in the calculation of b.
    """
    sup_vecs = support_vectors(ps)
    sum_f = lambda p: p.alpha*p.target*K(s,p.point)
    return sum(map(sum_f, sup_vecs))

def ind(s, ps, K, b):
    "The indicator function for vector s"
    return ind_no_b(s, ps, K) - b

def calc_b(ps, K):
    s, s_t, _ = next(support_vectors(ps))
    return ind_no_b(s, ps, K) - s_t

def main():
    global t, N, C
    N=10
    C=5
    x=[1,2,3,4,5,6,7,8,9,10]
    t=[1,2,3,4,5,6,7,8,9,10]
    initialize_P(x, t, K)
    start = numpy.zeros(N)
    ret = minimize(objective, start, bounds=[(0, C) for b in range(N)],
            constraints={'type':'eq', 'fun':zerofun})
    alpha = ret['x']

if __name__ == "__main__":
    main()
