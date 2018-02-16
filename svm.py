import numpy, random ,math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import namedtuple

global t
global P

def indicator():
    # Implement the indicator function (equation 6) which uses the non-zero αi
    #’s together with their ⃗xi
    #’s and ti
    #’s to classify new points.
    pass

def zerofun(vector):
    # calculates the value which should be constrained to zero.
    pass

def initalize_P(x, t, K):
    global P
    P = numpy.array([[t[i]*t[j]]*K(x[i], x[j]) for i in range(0,N)] for j in range(0,N)])

def objective(alfa):
    m = numpy.matrix([[alfa[i]*alfa[j]*P[i][j] for i in range(0,N)] for j in range(0,N)])
    n = numpy.array([alfa[i] for i in range(0,N))
    return m.sum()/2 - n.sum()

def K(a,b):
    return linear_kernel(a,b)

def linear_kernel(a, b):
    assert len(a) == len(b)
    return sum([a[i][0]*b[i] for i in range(len(b))])

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
    "Calculate the b value"
    s, s_t, _ = next(support_vectors(ps))
    return ind_no_b(s, ps, K) - s_t

def main():
    N=1
    start = numpy.zeros(N)
    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']

if __name__ == "__main__":
    main()
