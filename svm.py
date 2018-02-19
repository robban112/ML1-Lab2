import numpy, random ,math
from scipy.optimize import minimize
from collections import namedtuple
from numpy import linalg as LA

global t
global N

######### INTERNAL/PRIVATE FUNCTIONS #########

def zerofun(alfa):
    " Defines the upper bound to incorporate slack variables "
    return numpy.dot(alfa, t)

def objective(alfa, P):
    m = numpy.matrix([[alfa[i]*alfa[j]*P[i][j] for i in range(0,N)] for j in range(0,N)])
    n = numpy.array([alfa[i] for i in range(0,N)])
    return m.sum()/2 - n.sum()

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
    sum_f = lambda p: p.alpha*p.target*K(s,p.point)
    return sum(map(sum_f, ps))

def calc_b(ps, K):
    s, s_t, _ = ps[0]
    return ind_no_b(s, ps, K) - s_t

# Kernels
def linear_kernel(a, b):
    return numpy.dot(a,b)

def radial_kernel(sigma):
    def rad_kernel(x, y):
        diff = numpy.subtract(x,y)
        norm = LA.norm(diff)
        exp = -((norm**2) / (2*(sigma**2)))
        return math.exp(exp)
    return rad_kernel

def polynomial_kernel(p):
    def poly_kernel(a,b):
        return numpy.power(numpy.dot(a,b)+1, p)
    return poly_kernel

################# PUBLIC API #################

# A Support Vector Machine
SVM = namedtuple('SVM', ['ps','K','b'])

def train_svm(data_points, targets, kernel, C):
    "Train a Support Vector Machine using the given data, kernel and C value"
    global t, N
    x, t, K = (data_points, targets, kernel)
    N = len(x)
    start = numpy.zeros(N)
    P = numpy.array([[t[i]*t[j]*K(x[i], x[j]) for i in range(0,N)] for j in range(0,N)])
    ret = minimize(objective, start, args=(P), bounds=[(0, C) for b in range(N)],
            constraints={'type':'eq', 'fun':zerofun})
    if not ret.success:
        return None
    alpha = ret['x']
    ps = list(filter(is_support_vector, map(DataPointInfo._make, zip(x,t,alpha))))
    b = calc_b(ps, K)
    return SVM(ps=ps, K=K, b=b)

def ind(s, svm):
    "The indicator function for vector s"
    return ind_no_b(s, svm.ps, svm.K) - svm.b
