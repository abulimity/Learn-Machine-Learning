import numpy as np
import random
import logging
import pandas as pd


class Perceptron(object):

    def __init__(self, eta=0.00001, max_iter=5000):
        self.eta = eta
        self.max_iter = max_iter
        self.w = 0

    def fit(self, X, y):
        X = np.array(pd.DataFrame(X).to_numpy())
        y = np.array(pd.DataFrame(y).to_numpy())
        self.w = np.zeros(X.shape[1]+1)
        n_iter = 0
        while n_iter <= self.max_iter:
            index = random.randint(0, len(y)-1)
            x = np.hstack([X[index], 1])
            y_ = 2 * y[index] - 1

            yHat = np.dot(self.w, x)
            loss = y_ * yHat
            if loss <= 0:
                self.w += self.eta * y_ * x
            n_iter += 1
            # logging.info('iter: %s' % n_iter)

    def predict(self, X):
        X = np.hstack([X, np.ones(shape=X.shape[0]).reshape(-1, 1)])
        y = [1 if y else 0 for y in np.dot(X, self.w) > 0]
        return np.array(y)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # logging.info('ddddd')
