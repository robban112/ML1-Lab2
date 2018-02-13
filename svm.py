import numpy, random ,math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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


def main():
    N=1
    start = numpy.zeros(N)
    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']

if __name__ == "__main__":
    main()
