import numpy as np
import sys
from helper import *


def third_order(X):
        """Third order polynomial transform on features X.

        Args:
                X: An array with shape [n_samples, 2].

        Returns:
                poly: An (numpy) array with shape [n_samples, 10].
        """
        poly = np.zeros((X[:,0].size,10))
    ### YOUR CODE HERE
        for i,x in enumerate(X):
                x_1 = X[i,0]
                x_2 = X[i,1]
                transformed = np.array([1,x_1,x_2,x_1**2,x_1*x_2,x_2**2,x_1**3,x_1**2 * 2,x_1 * x_2**2,x_2**3])
                poly[i] = transformed
        return poly
        ### END YOUR CODE


class LogisticRegression(object):
        
        def __init__(self, max_iter, learning_rate, third_order=False):
                self.max_iter = max_iter
                self.lr = learning_rate
                self.third_order = third_order


        def _gradient(self, X, y):
                """Compute the gradient with samples (X, y) and weights self.W.

                Args:
                        X: An array with shape [n_samples, n_features].
                           (n_features depends on whether third_order is applied.)
                        y: An array with shape [n_samples,]. Only contains 1 or -1.

                Returns:
                        gradient: An array with shape [n_features,].
                """
                ### YOUR CODE HERE
                X = np.array(X)
                y = np.array(y)
                gradient = 0
                for i,x in enumerate(X):
                    ###numerator = []
                    ###for x in X[i]:
                    ###      numerator.append(y[i] * x)
                    ###denominator = 1 + np.exp(y[i] * np.dot(self.W,X[i]))
                    ###temp = numerator/denominator * -1/self.max_iter
                    ###gradient = gradient + temp
                    z = np.dot(X[i],self.W)
                    h = 1 / (1 + np.exp(-z))
                    gradient = np.dot(X.T, (h - y)) / y.size
                return gradient
                ### END YOUR CODE


        def fit(self, X, y):
                """Train logistic regression model on data (X,y).
                (If third_order is true, do the 3rd order polynomial transform)

                Args:
                        X: An array with shape [n_samples, 3].
                        y: An array with shape [n_samples,]. Only contains 1 or -1.

                Returns:
                        self: Returns an instance of self.
                """
            ### YOUR CODE HERE
                if self.third_order:
                    self.W = np.ones(10)
                    X = third_order(X)
                else:
                    self.W = np.ones(3)
                
                for t in range(self.max_iter):
                    gradient = self._gradient(X,y)
                    gradient = np.array(gradient)
                    self.W -= self.lr * gradient
            ### END YOUR CODE
                return self


        def get_params(self):
                """Get parameters for this perceptron model.

                Returns:
                        W: An array of shape [n_features,].
                           (n_features depends on whether third_order is applied.)
                """
                if self.W is None:
                        print("Run fit first!")
                        sys.exit(-1)
                return self.W


        def predict(self, X):
                """Predict class labels for samples in X.
                (If third_order is true, do the 3rd order polynomial transform)

                Args:
                        X: An array of shape [n_samples, 3].

                Returns:
                        preds: An array of shape [n_samples,]. Only contains 1 or -1.
                """
                ### YOUR CODE HERE
                preds = []
                if self.third_order:
                    X = third_order(X)
                for i, x in enumerate(X):
                    z = np.dot(X[i],self.W)
                    sigmoid = 1 / (1 + np.exp(-z))
                    if sigmoid >= 0.5:
                        preds.append(1)
                    else:
                        preds.append(-1)
                return preds
                ### END YOUR CODE


        def score(self, X, y):
                """Returns the mean accuracy on the given test data and labels.

                Args:
                        X: An array of shape [n_samples, n_features].
                        y: An array of shape [n_samples,]. Only contains 1 or -1.

                Returns:
                        score: A float. Mean accuracy of self.predict(X) wrt. y.
                """
                return np.mean(self.predict(X)==y)



def accuracy_logreg(max_iter, learning_rate, third_order, 
                                        X_train, y_train, X_test, y_test):

        # train perceptron
        model = LogisticRegression(max_iter, learning_rate, third_order)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)

        # test perceptron model
        test_acc = model.score(X_test, y_test)

        return train_acc, test_acc
