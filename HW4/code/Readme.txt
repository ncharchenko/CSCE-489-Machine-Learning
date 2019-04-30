
LIBSVM tutorial
---------------
the libsvm library has been put into the code folder and already been imported to the 'solution.py' file. You can directly call the corresponding functions in 'solution.py' file. In this homework, you will need two functions name: 'svm_train' and 'svm_predict' to start your work. Following is how it works.

1. Library setup

In the command line, change to location './libsvm/' and run 'make' command, then to location './libsvm/' and run 'make' command again. The installation will finish automatically. Then you will be able use certain functions in the libsvm package.


2. To train a svm classifier:

In Python, command:
model = svm_train(train_label, train_data, libsvm_options)

- train_label: an m by 1 list of training labels. m represents total training data samples.
- train_data: an m by n two dimension list. m represents total training data samples and n represents the number of features for each data sample.
- libsvm_options: a string format of training options. You will be using the following options in your homework:
-c cost: set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-t kernel: set type of kernel function (default 3). 0: linear kernel; 1: polynomial kernel; 2: radial basis function kernel.

Here is an example to use svm_train:

suppose you have train_data and train_label, then set up libsvm options as:
libsvm_options = '-c 2 -t 1' and code:
model = svm_train(train_label, train_data, libsvm_options)

You will get a svm model with polynomial kernel saved on 'model' and it will also print following information:
**********************************
optimization finished, #iter = 254
nu = 0.322870
obj = -34.870094, rho = -0.069467
nSV = 504, nBSV = 504
Total nSV = 504
**********************************************

Where #iter means the total iterations to find the optimal solution; Total nSV means the total number of support vectors in this model.

3. Predict label on test data

Command:
predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)

- test_label: an m by 1 list of prediction labels
- test_data: an m by n two dimension list. m represents total testing data samples, and n represents the number of features for each data sample.
- model: the output of the svm_train function.
This function will return three values: predicted label, test accuracy and decision values with classification accuracy printing out.

Following is an example printing output after call this function:
**********************************************
Accuracy = 96.2264% (408/424) (classification)
**********************************************



Free feel to email Yaochen Xie for any assistance.
Email address: ethanycx@tamu.edu.
