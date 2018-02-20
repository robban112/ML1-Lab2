from svm import train_svm, ind, linear_kernel, radial_kernel, polynomial_kernel
import testdata_generator as test_gen
import matplotlib.pyplot as plt
import numpy

def plot(classA, classB, svm):
    "Plot the bounderies"
    plt.plot([p[0] for p in classA], [p[1] for p in classA],'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB],'r.')
    plt.axis('equal')
    xgrid=numpy.linspace(-5,5)
    ygrid=numpy.linspace(-4,4)
    grid=numpy.array([[ind((x,y), svm) for x in xgrid ] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors =('red' ,'black' , 'blue'),
                                                              linewidths=(1 , 3 , 1))
    plt.show()

def main():
    # 1. generate the training data
    classA, classB = test_gen.test_classes_3()
    data_points = test_gen.generate_input_data(classA, classB)
    x = [p.coords for p in data_points]
    t = [p.target for p in data_points]

    # 2. train the SVM
    K = radial_kernel(sigma=0.4)
    svm = train_svm(data_points=x, targets=t, kernel=K, C=20)

    # 3. plot the SVM bounderies
    plot(classA, classB, svm)

if __name__ == "__main__":
    main()
