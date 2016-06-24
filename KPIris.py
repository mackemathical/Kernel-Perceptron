import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


# Below, we have the KernelPerceptron algorithm, as before.
class KernelPerceptron(object):
    def __init__(self, n=10, kernel=None, sigma=1.0, theta=0.0, p=1.0, eta=1.0, c1=0.0, c2=1.0):
        self.n = n  # iterations
        self.kernel = kernel  # the kernel we use. By default, we just take a dot product.
        self.sigma = sigma  # sigma for RBF kernel
        self.theta = theta  # theta for polynomial and sigmoid kernels
        self.p = p  # power for polynomial kernel
        self.eta = eta  # eta for sigmoid kernel
        self.c1 = c1  # addition c for generalized kernel function
        self.c2 = c2  # multiplication c for generalized kernel function

    def fit(self, x, y):
        self.a_ = np.zeros(x.shape[0])
        self.errors_ = []
        self.x_ = x
        self.y_ = y

        for iteration in range(self.n):
            errors = 0
            for j in range(self.a_.size):
                if self.predict(self.x_.iloc[j, :]) != y[j]:  # we only update when we make mistakes
                    self.a_[j] += 1  # our whole update rule is just keeping track of errors!
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                print('Converged after %.i iterations.' % iteration)
                break  # if we don't update in an iteration, it saves time to just stop, and let the user know.
        return self

    def predict(self, xj):
        total = 0  # our predict function can be written sign(sum(a_i*y_i*K(x_i,x_j)))
        for i in range(self.a_.size):
            total += self.a_[i]*self.y_[i]*self.kernel_function(self.x_.iloc[i, :], xj)
        return np.where(total >= 0.0, 1, -1)  # np.where conveniently gives a value for 0.

    def kernel_function(self, xi, xj):
        if self.kernel is None:  # if we're not using a kernel, we just use a dot product.
            kern = np.dot(xi, xj)
        elif self.kernel == 'rbf' or self.kernel == 'Gaussian':
            num = np.linalg.norm(xi-xj)**2
            den = 2*self.sigma**2
            frac = num / den
            kern = np.exp(-1 * frac)  # this is the definition of the RBF kernel.
        elif self.kernel == 'polynomial':
            kern = (np.dot(xi, xj) + self.theta)**self.p  # this is the definition of the polynomial kernel.
        else:
            raise ValueError('Unrecognized kernel "' + self.kernel + '"')  # mainly for typos and whatnot
        return self.c1 + self.c2 * kern  # user-specified c1 and c2 allow us to generalize the kernel function

    def score(self, x, y):
        correct = 0
        total = x.shape[0]
        for j in range(total):
            if self.predict(x.iloc[j, :]) == y[j]:
                correct += 1
        return correct / total  # self-explanatory: returns a % of correct answers


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = pd.DataFrame(df.iloc[0:100, [0, 2]].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#We start by importing the Iris data for a quick test that we can converge

kp = KernelPerceptron()
print('Standard Perceptron:')
kp.fit(X_train, y_train)  # this returns the string 'converged after 1 iteration.' Not bad!
kp.score(X_train, y_train)
kp.score(X_test, y_test)  # you'll notice we do perfect on the test data as well.

kpk = KernelPerceptron(kernel='rbf')  # now we test it with the RBF kernel
print('RBF Kernel')
kpk.fit(X_train, y_train)  # this returns the same string as above, how impressive!
kpk.score(X_train, y_train)
kpk.score(X_test, y_test)  # and, once again, it scores perfectly on linearly separable data.

kpk2 = KernelPerceptron(kernel='polynomial', theta=7, p=3.4, c1=3.14159, c2=4.7)  # test some crazy parameters
print('Crazy Polynomial:')
kpk2.fit(X_train, y_train)  # this time it takes 2 iterations to converge, showing some kernels are better than others.
kpk2.score(X_train, y_train)
kpk2.score(X_test, y_test)  # as expected, any converged perceptron should give 100% accuracy.
