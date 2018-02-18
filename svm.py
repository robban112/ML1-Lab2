import numpy, random ,math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import namedtuple
import testdata_generator as test_gen

global t
global N
global C

def zerofun(alfa):
    " Defines the upper bound to incorporate slack variables "
    return numpy.dot(alfa, t)

def objective(alfa, P):
    m = numpy.matrix([[alfa[i]*alfa[j]*P[i][j] for i in range(0,N)] for j in range(0,N)])
    n = numpy.array([alfa[i] for i in range(0,N)])
    return m.sum()/2 - n.sum()

# def objective(alfa):
#     return sum(numpy.dot(alfa,P))/2 - numpy.sum(alfa)

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
    sum_f = lambda p: p.alpha*p.target*K(s,p.point)
    return sum(map(sum_f, ps))

def ind(s, ps, K, b):
    "The indicator function for vector s"
    return ind_no_b(s, ps, K) - b

def calc_b(ps, K):
    s, s_t, _ = ps[0]
    return ind_no_b(s, ps, K) - s_t

def plot(classA, classB, ps, b):
    plt.plot([p[0] for p in classA], [p[1] for p in classA],'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB],'r.')
    plt.axis('equal')
    xgrid=numpy.linspace(-5,5)
    ygrid=numpy.linspace(-4,4)
    grid=numpy.array([[ind((x,y), ps, K, b) for y in ygrid ] for x in xgrid])
    plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors =('red' ,'black' , 'blue'),
                                                              linewidths=(1 , 3 , 1))
    plt.show()


def main():
    global t, N, C
    # N=10
    C=20
    classA, classB = test_gen.test_classes_1()
    data_points = test_gen.generate_input_data(classA, classB)
    x = [p.coords for p in data_points]
    t = [p.target for p in data_points]
    N = len(data_points)
    start = numpy.zeros(N)
    P = numpy.array([[t[i]*t[j]*K(x[i], x[j]) for i in range(0,N)] for j in range(0,N)])
    ret = minimize(objective, start, args=(P), bounds=[(0, C) for b in range(N)],
            constraints={'type':'eq', 'fun':zerofun})
    if not ret.success:
        print("Optimization was unsuccessful")
        return
    alpha = ret['x']
    ps = list(filter(is_support_vector, map(DataPointInfo._make, zip(x,t,alpha))))
    b = calc_b(ps, K)
    plot(classA, classB, ps, b)

if __name__ == "__main__":
    main()
