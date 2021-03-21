import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from perceptron import Perceptron
# import logging


class TestPerceptron(unittest.TestCase):
    def test_a(self):
        dataSet = pd.read_csv(r'data\CH05_Perceptron\testSet.txt', sep=',', engine='python', header=None, names=['x1', 'x2', 'y'])
        # dataSet['y'][dataSet.y == 0] = -1
        X_train, X_test, y_train, y_test = train_test_split(dataSet.loc[:, ['x1', 'x2']], dataSet['y'], train_size=0.5)
        # logging.warning(X_test)
        clf = Perceptron()
        clf.fit(X_train, y_train)
        result = pd.Series(clf.predict(X_test))
        print('result: %s' % result.to_numpy())
        print('result: %s' % y_test.to_numpy())
        assert (result.to_numpy() == y_test.to_numpy()).all()


if __name__ == '__main__':
    unittest.main(verbosity=0)
