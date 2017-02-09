from Term_1.MinFlow.MiniFlow_challenge.MiniFLow import *
from Term_1.MinFlow.MiniFlow_challenge.mnist import *
from sklearn.utils import shuffle, resample
import numpy as np

np.set_printoptions(threshold=np.nan)


def run():
    """
    Check out the new network architecture and dataset!

    Notice that the weights and biases are
    generated randomly.

    No need to change anything, but feel free to tweak
    to test your network, play around with the epochs, batch size, etc!
    """

    # Load data
    data = read()
    training = data['training']
    testing = data['testing']
    temp_training = []
    temp_testing = []

    for _x in training:
        temp_training.append(_x.reshape(-1))

    for _y in testing:
        temp_testing.append(_y.flatten())

    X_ = np.asarray(temp_training, dtype=np.float)
    y_ = np.asarray(temp_testing, dtype=np.float)

    cal = (X_ - np.mean(X_, axis=0))/np.std(X_, axis=0)
    cal = np.nan_to_num(cal)
    # print(np.std(X_, axis=0))

    # Normalize data
    X_ = cal
    #
    n_features = X_.shape[1]
    n_hidden = 10
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)
    #
    # Neural network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    epochs = 10
    # Total number of examples
    m = X_.shape[0]
    batch_size = 11
    steps_per_epoch = m // batch_size

    math_stuff = MathStuff()

    graph = math_stuff.topological_sort(feed_dict)
    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    # Step 4
    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # Step 1
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            # Reset value of X and y Inputs
            X.value = X_batch
            y.value = y_batch

            # Step 2
            math_stuff.forward_and_backward(graph)

            # Step 3
            math_stuff.sgd_update(trainables)

            loss += graph[-1].value

        print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))

if __name__ == '__main__':
    run()
