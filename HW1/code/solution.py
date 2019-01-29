import numpy as np
import sys
from helper import *


def show_images(data):
        """Show the input images and save them.

        Args:
                data: A stack of two images from train data with shape (2, 16, 16).
                          Each of the image has the shape (16, 16)

        Returns:
                Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
                include them in your report
        """
        ### YOUR CODE HERE
        print("Image 1")
        #plt.imshow(data[0])
        plt.imsave('image_1.png',data[0])
        print("Image 2")
        #plt.imshow(data[1])
        plt.imsave('image_2.png',data[1])
        ### END YOUR CODE


def show_features(X, y, save=True):
        """Plot a 2-D scatter plot in the feature space and save it. 

        Args:
                X: An array of shape [n_samples, n_features].
                y: An array of shape [n_samples,]. Only contains 1 or -1.
                save: Boolean. The function will save the figure only if save is True.

        Returns:
                Do not return any arguments. Save the plot to 'train_features.*' and include it
                in your report.
        """
        ### YOUR CODE HERE
        plt.scatter(X[:,0],X[:,1],c=y)
        plt.title("Features Scatter Plot")
        plt.xlabel('x-features')
        plt.ylabel('y-features')
        if save == True:
            plt.savefig('train_features.png')
        plt.show()
        ### END YOUR CODE


class Perceptron(object):
        
        def __init__(self, max_iter):
                self.max_iter = max_iter

        def fit(self, X, y):
                """Train perceptron model on data (X,y).

                Args:
                        X: An array of shape [n_samples, n_features].
                        y: An array of shape [n_samples,]. Only contains 1 or -1.

                Returns:
                        self: Returns an instance of self.
                """
                ### YOUR CODE HERE
                self.W = np.zeros(len(X[0]))

                for t in range(self.max_iter):
                    for i, x in enumerate(X):
                        if (np.dot(self.W,X[i])*y[i]) <= 0:
                            self.W = self.W + X[i]*y[i]
                

                ### END YOUR CODE
                
                return self

        def get_params(self):
                """Get parameters for this perceptron model.

                Returns:
                        W: An array of shape [n_features,].
                """
                if self.W is None:
                        print("Run fit first!")
                        sys.exit(-1)
                return self.W

        def predict(self, X):
                """Predict class labels for samples in X.

                Args:
                        X: An array of shape [n_samples, n_features].

                Returns:
                        preds: An array of shape [n_samples,]. Only contains 1 or -1.
                """
                ### YOUR CODE HERE
                preds = []
                for i, x in enumerate(X):
                    preds.append(np.sign(np.dot(self.W,X[i])))
                return preds
                ### END YOUR CODE

        def score(self, X, y):
                """Returns the mean accuracy on the given test data and labels.

                Args:
                        X: An array of shape [n_samples, n_features].
                        y: An array of shape [n_samples,]. Only contains 1 or -1.

                Returns:
                        score: An float. Mean accuracy of self.predict(X) wrt. y.
                """
                ### YOUR CODE HERE
                num_samples = len(y)
                correct_preds = 0
                preds = self.predict(X)
                for i,x in enumerate(X):
                    if preds[i] == y[i]:
                        correct_preds += 1
                return correct_preds/num_samples
                ### END YOUR CODE




def show_result(X, y, W):
        """Plot the linear model after training. 
           You can call show_features with 'save' being False for convenience.

        Args:
                X: An array of shape [n_samples, 2].
                y: An array of shape [n_samples,]. Only contains 1 or -1.
                W: An array of shape [n_features,].
        
        Returns:
                Do not return any arguments. Save the plot to 'result.*' and include it
                in your report.
        """
        ### YOUR CODE HERE
        x = len(X[0])
        formula = "(-(W[0] / W[2]) / (W[0] / W[1]))*x + (-W[0] / W[2])"
        Y = eval(formula)
        show_features(X,y,False)
        plt.plot(x,Y)
        plt.show()
        
        ### END YOUR CODE



def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

        # train perceptron
        model = Perceptron(max_iter)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        W = model.get_params()

        # test perceptron model
        test_acc = model.score(X_test, y_test)

        return W, train_acc, test_acc
