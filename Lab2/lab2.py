from cvxopt.solvers import qp
from cvxopt.base import matrix
import random, pylab, math
import numpy as np

classA = []
classB = []
data = []
epsilon = 0.0001


def get_data():
    global classA, classB, data
    np.random.seed(100)
    classA = [(random.normalvariate(1.5, 1.0), random.normalvariate(0.5, 1.0), 1.0) for i in range(5)] + \
             [(random.normalvariate(1.5, 1.0), random.normalvariate(0.5, 1.0), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(0.5, 0.5), -1.0) for i in range(10)]
    data = classA + classB
    random.shuffle(data)
    return data

def scalar_product(x, y, dim):
    product = 0.0
    for i in range(dim):
        product += (x[i] * y[i])
    return product

def linear_kernel(x, y):
    linear = scalar_product(x, y, 2) + 1.0
    return linear

def poly_kernel(x, y, degree):
    poly = math.pow(linear_kernel(x, y), degree)
    return poly

def rbf_kernel(x, y, sigma):
    i = (x[0] - y[0], x[1] - y[1])
    rbf = math.exp(-1 * (scalar_product(i, i, 2) / (2 * sigma * sigma)))
    return rbf

def sigmoid_kernel(x, y, k, delta):
    sigmoid = math.tanh(np.dot(np.multiply(k, x), y) + delta)
    return sigmoid

def p_matrix(data, kernel_function, **keys):
    pmatrix = np.matrix(np.zeros((len(data), len(data))))
    for i in list(range(len(data))):
        for j in list(range(len(data))):
            if (keys):
                kernel = kernel_function(data[i][:2], data[j][:2], **keys)
            else:
                kernel = kernel_function(data[i][:2], data[j][:2])
            pmatrix[i, j] = data[i][2] * data[j][2] * kernel
    return pmatrix

def get_support_vectors(data, alphas):
    support_vectors = []
    for i in list(range(len(data))):
        if abs(alphas[i]) > epsilon:
            support_vectors.append((alphas[i], data[i]))
    return support_vectors

def ind_function(x, support_vectors, kernel_function, **params):
    indicator = 0
    for i in range(len(support_vectors)):
        alpha, y = support_vectors[i]
        if (params):
            indicator += alpha * y[2] * kernel_function(x, y, **params)
        else:
            indicator += alpha * y[2] * kernel_function(x, y)
    return indicator

def svm():
    data = get_data()
    kernel_function = rbf_kernel
    slack = True
    C = 1

    # Solve function overloading with params
    if (kernel_function == poly_kernel):
        params = dict(degree=8)
    elif (kernel_function == rbf_kernel):
        params = dict(sigma=100)
    elif (kernel_function == sigmoid_kernel):
        params = dict(k=0.001, delta=0.1)
    else:
        params = {}

    # Create matrices
    p = matrix(p_matrix(data, kernel_function, **params))
    q = matrix(-1.0, (len(data), 1))
    print(p.size)

    if (not slack):
        g = matrix(np.diag([-1.0] * len(data)))
        h = matrix(0.0, (len(data), 1))
    else:
        g = matrix(np.concatenate((np.diag([-1.0] * len(data)), np.diag([1.0] * len(data)))))   # axis=0
        h = matrix(np.concatenate(([0.0] * len(data), [1.0 * C] * len(data))))   #axis=1

    r = qp(p, q, g, h, kktsolver='ldl')      # kktsolver='ldl'
    alphas = list(r['x'])
    #print(type(g))
    #print(alphas)
    support_vectors = get_support_vectors(data, alphas)
    #print(len(support_vectors))

    pylab.figure()
    title = kernel_function
    title2 = C
    title3 = slack
    title4 = " kernel: " + str(title) +"slack" + str(title3) + "C: " + str(title2)
    pylab.title(str(title4))
    pylab.plot([p[0] for p in classA],
               [p[1] for p in classA],
               'bo')
    pylab.plot([p[0] for p in classB],
               [p[1] for p in classB],
               'ro')

    pylab.plot([p[0] for (a, p) in support_vectors],
               [p[1] for (a, p) in support_vectors],
               'kx')

    x_range = np.arange(-4, 4, 0.05)
    y_range = np.arange(-4, 4, 0.05)
    grid = matrix([[ind_function((x, y), support_vectors, kernel_function, **params)
                    for y in y_range]
                   for x in x_range])
    pylab.contour(x_range, y_range, grid,
                  (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'),
                  linewidths=(1, 3, 1))
    #pylab.savefig('' + 'tryout.png', dpi=400)
    pylab.show()

def main():
    svm()

if __name__ == '__main__':
    main()


