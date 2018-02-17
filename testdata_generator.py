import numpy
from random import shuffle
from collections import namedtuple

"""Module usage:
 1. First call one of the `test_classes_x` functions, to get data points divided
into two classes, a class A and a class B.
 2. Then to get the actual input data to the svm, pass the two classes received from
the previous step into the function `generate_input_data`. This will assign a target
value of 1 to the points in class A and -1 to points in class B, and return all the
points in random order.
"""

# A data point, with coordinates and a target value (-1 or 1)
Datapoint = namedtuple('Datapoint', ['coords','target'])

def norm_points(N, mu_x, mu_y, sigma):
    """Get N points with x-values sampled from a N(mu_x, sigma^2) distribution
    and y-values sampled from a N(mu_y, sigma^2) distribution.
    """
    return sigma * numpy.random.randn(N, 2) + [mu_x,mu_y]

def generate_input_data(classA, classB):
    """Assign a target value of 1 to the points of class A, and -1 to the
    points of class B. Concatenate all the data points, and shuffle.
    """
    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate((
        numpy.ones(classA.shape[0]),
        -numpy.ones(classB.shape[0])
    ))
    points = list(map(Datapoint._make, zip(inputs, targets)))
    shuffle(points)
    return points

############ The different test classes to try out ############
def test_classes_1():
    "Get data points divided into two classes, class A and B."
    classA = numpy.concatenate((
        norm_points(N=10,mu_x=1.5,mu_y=0.5,sigma=0.2),
        norm_points(N=10,mu_x=-1.5,mu_y=0.5,sigma=0.2)
    ))
    classB = norm_points(N=20,mu_x=0,mu_y=-0.5,sigma=0.2)
    return (classA, classB)

def test_classes_2():
    "Get data points divided into two classes, class A and B."
    classA = numpy.concatenate((
        norm_points(N=10,mu_x=1.5,mu_y=1.5,sigma=0.3),
        norm_points(N=10,mu_x=-1.5,mu_y=-1.5,sigma=0.3)
    ))
    classB = norm_points(N=20,mu_x=0,mu_y=0,sigma=0.3)
    return (classA, classB)

def test_classes_3():
    "Get data points divided into two classes, class A and B."
    classA = numpy.concatenate((
        norm_points(N=10,mu_x=-1.5,mu_y=0,sigma=0.2),
        norm_points(N=10,mu_x=0.5,mu_y=0,sigma=0.2)
    ))
    classB = numpy.concatenate((
        norm_points(N=10,mu_x=-0.5,mu_y=0,sigma=0.2),
        norm_points(N=10,mu_x=1.5,mu_y=0,sigma=0.2)
    ))
    return (classA, classB)
