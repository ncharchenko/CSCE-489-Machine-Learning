from helper import *
from solution import *



def test_logreg(X_train, y_train, X_test, y_test, third_order):
	max_iters = [100, 200, 500,1000]
	learning_rates = [0.1, 0.2, 0.5]

	for i, m_iter in enumerate(max_iters):
		train_acc, test_acc = accuracy_logreg(m_iter, learning_rates[1], 
											  third_order, X_train, y_train, 
											  X_test, y_test)
		print('Max iteration testcase %d: Max iter: %d Train accuracy: %f,\
			   Test accuracy: %f'%(i, m_iter, train_acc, test_acc))

	for i, learning_rate in enumerate(learning_rates):
		train_acc, test_acc = accuracy_logreg(max_iters[3], learning_rate, 
											  third_order, X_train, y_train, 
											  X_test, y_test)
		print('Learning rate testcase %d: Learning rate: %.2f Train accuracy: %f,\
			   Test accuracy: %f'%(i, learning_rate, train_acc, test_acc))


if __name__ == '__main__':

	traindataloc = "../data/train.txt"
	testdataloc = "../data/test.txt"
	data_train = load_data(traindataloc)
	X_train, y_train = load_features(traindataloc)
	X_test, y_test = load_features(testdataloc)

	print('Testing with Linear Logistic Regression: ')
	test_logreg(X_train, y_train, X_test, y_test, False)

	print('Testing with 3rd order polynomial Logistic Regression: ')
	test_logreg(X_train, y_train, X_test, y_test, True)