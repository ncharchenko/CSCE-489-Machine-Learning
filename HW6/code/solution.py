import tensorflow as tf
import numpy as np
import pickle, tqdm, os
import time

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_dir):
    '''
    To load the Cifar-10 Dataset from files and reshape the 
    images arrays from shape [3072,] to shape [32, 32, 3].

    Please follow the instruction on how to load the data and 
    labels at https://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        data_dir: String. The directory where data batches are 
            stored.

    Returns:
        x_train: An numpy array of shape [50000, 32, 32, 3].
            (dtype=np.uint8)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int64)
        x_test: An numpy array of shape [10000, 32, 32, 3].
            (dtype=np.uint8)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int64)
    '''
    negatives = False
    meta_data_dict = unpickle(data_dir + "batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    x_train = []
    train_filenames = []
    y_train = []

    for i in range(1, 6):
        train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            x_train = train_data_dict[b'data']
        else:
            x_train = np.vstack((x_train, train_data_dict[b'data']))
        train_filenames += train_data_dict[b'filenames']
        y_train += train_data_dict[b'labels']

    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    if negatives:
        x_train = x_train.transpose(0, 2, 3, 1).astype(np.uint8)
    else:
        x_train = np.rollaxis(x_train, 1, 4)
    train_filenames = np.array(train_filenames)
    y_train = np.array(y_train).astype(np.uint64)

    test_data_dict = unpickle(data_dir + "/test_batch")
    x_test = test_data_dict[b'data']
    test_filenames = test_data_dict[b'filenames']
    y_test = test_data_dict[b'labels']

    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    if negatives:
        x_test = x_test.transpose(0, 2, 3, 1).astype(np.uint8)
    else:
        x_test = np.rollaxis(x_test, 1, 4)
    test_filenames = np.array(test_filenames)
    y_test = np.array(y_test).astype(np.uint64)

    return x_train, y_train, x_test, y_test


def preprocess(train_images, test_images, normalize=False):
    '''
    To preprocess the data by 
        (1).Rescaling the pixels from integers in [0,255) to 
            floats in [0,1), or 
        (2).Normalizing each image using its mean and variance. 

    If you are working on the honor section, please implement 
        (1) and then (2). 
    If not, please implement (1) only.

    Args:
        train_images: An numpy array of shape [50000, 32, 32, 3].
            (dtype=np.uint8)
        test_images: An numpy array of shape [10000, 32, 32, 3].
            (dtype=np.uint8)
        normalize: Boolean. To control to rescale or normalize 
            the images. (Only for the honor section)

    Returns:
        train_images: An numpy array of shape [50000, 32, 32, 3].
            (dtype=np.float64)
        test_images: An numpy array of shape [10000, 32, 32, 3].
            (dtype=np.float64)
    '''
    train_images = train_images * (1/255)
    test_images = test_images * (1/255)
    return train_images, test_images


class LeNet_Cifar10():
    def __init__(self, sess, n_classes):
        '''
        Args:
            sess: tf.Session. A Tensorflow session.
            n_classes: Integer. The number of classes.

        Attributes:
            X: Tensor of shape [None, 32, 32, 3]. A placeholder 
                for input images. "None" refers to any batch size.
            y: Tensor of shape [None]. A placeholder for input
                labels. "None" refers to any batch size.
            training: (used in the honor section)
                Tensor of boolean. A placeholder to specify
                training phrase or not for Dropout and Batchnorm.
            logits: Tensor of shape [None, n_classes]. Output 
                signal of neural network.
        '''
        self.sess = sess
        self.n_classes = n_classes
        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None,))
        self.training = tf.placeholder(tf.bool)
        self.logits = self._network(self.X)

    def _network(self, X):
        '''
        To build the graph of LeCun network:
            Inputs --> 
            Convolution (6) --> Max Pooling --> 
            Convolution (16) --> Max Pooling --> 
            Reshape to vector --> 
            Fully-connected (120) --> 
            Fully-connected (84) --> Outputs (n_classes).

        You are free to use the listed APIs from tf.layers or tf.nn:
            tf.layers.conv2d
            tf.layers.max_pooling2d
            tf.layers.flatten
            tf.layers.dense
            tf.nn.relu (or other activations)

        For the honor section, you may also need:
            tf.layers.dropout
            tf.layers.batch_normalization

        Refer to https://www.tensorflow.org/api_docs/python/tf
        for the instructions for those APIs
        
        Args:
            X: Tensor. A placeholder of shape [None, 32, 32, 3]
                for input images.

        Returns:
            logits: Tensor of shape [None, n_classes].
        '''
        o1 = tf.layers.conv2d(inputs=X,filters=6,kernel_size=(5,5),)
        o1 = tf.nn.relu(o1)
        o2 = tf.layers.max_pooling2d(o1, (2,2), 2)
        o3 = tf.layers.conv2d(inputs=o2,filters=16,kernel_size=(5,5),)
        o3 = tf.nn.relu(o3)
        o4 = tf.layers.max_pooling2d(o3, (2,2), 2)
        o5 = tf.layers.flatten(o4)
        o6 = tf.layers.dense(o5,120)
        o6 = tf.nn.relu(o6)
        o7 = tf.layers.dense(o6,84)
        o7 = tf.nn.relu(o7)
        o8 = tf.layers.dense(o7, self.n_classes)
        logits = tf.nn.softmax(o8)
        return logits

    def _setup(self):
        '''
        Model and training setup.

        Attributes:
            preds: Tensor of shape [n_batch,]. Predicted classes of
                the given batch.
            correct: Tensor of shape [1,]. Number of correct prediction
                at the current batch.
            accuracy: Tensor of shape [1,]. Prediction accuracy at
                the current batch.
            loss: Tensor of shape [1,]. Cross-entropy loss computed on
                the current batch.
            optimizer: tf.train.Optimizer. The optimizer for training
                the model. Different optimizers use different gradient
                descend policies.
            train_op: An Operation that updates the variables.
        '''

        self.preds = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
        self.correct = tf.reduce_sum(
            tf.cast(tf.equal(self.preds, self.y), tf.float32))
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.preds, self.y), tf.float32))
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y, logits=self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss)

    def train(self, x_train, y_train, x_valid, y_valid, batch_size, max_epoch):

        self._setup()
        self.sess.run(tf.global_variables_initializer())

        num_samples = x_train.shape[0]
        num_batches = int(num_samples / batch_size)

        num_valid_samples = x_valid.shape[0]
        num_valid_batches = (num_valid_samples - 1) // batch_size + 1

        print('---Run...')
        for epoch in range(1, max_epoch + 1):

            # To shuffle the data at the beginning of each epoch.
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # To start training at current epoch.
            loss_value = []
            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                batch_start_time = time.time()

                start = batch_size * i
                end = batch_size * (i + 1)
                x_batch = curr_x_train[start:end]
                y_batch = curr_y_train[start:end]

                ################################################################
                # For the honor section, you may need to modify feed_dict below.
                ################################################################
                feed_dict = {self.X: x_batch, 
                             self.y: y_batch}
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_op],
                    feed_dict=feed_dict)
                if not i % 10:
                    qbar.set_description(
                        'Epoch {:d} Loss {:.6f} Acc {:.4f}'.format(
                            epoch, loss, acc))

            # To start validation at the end of each epoch.
            correct = 0
            print('Doing validation...', end=' ')
            for i in range(num_valid_batches):

                start = batch_size * i
                end = min(batch_size * (i + 1), x_valid.shape[0])
                x_valid_batch = x_valid[start:end]
                y_valid_batch = y_valid[start:end]

                ################################################################
                # For the honor section, you may need to modify feed_dict below.
                ################################################################
                feed_dict = {self.X: x_valid_batch, 
                             self.y: y_valid_batch}
                correct += self.sess.run(self.correct, feed_dict=feed_dict)

            acc = correct / num_valid_samples
            print('Validation Acc {:.4f}'.format(acc))

    def test(self, X_test, y_test):

        accs = 0
        for X, y in zip(X_test, y_test):
            
            ################################################################
            # For the honor section, you may need to modify feed_dict below.
            ################################################################
            feed_dict={
                self.X: np.array([X]),
                self.y: np.array([y])}
            acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
            accs += acc

        accuracy = float(accs) / len(y_test)
        
        return accuracy
