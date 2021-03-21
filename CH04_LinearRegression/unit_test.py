from lr import LR
import unittest
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TestLR(unittest.TestCase):
    def test(self):
        dataSet = pd.read_csv(r'data\CH04_LinearRegression\ex0.txt', sep='\\t', engine='python', header=None, names=['x0', 'x1', 'y'])
        X_train, X_test, y_train, y_test = train_test_split(dataSet.loc[:, ['x0', 'x1']], dataSet['y'], train_size=0.8)
        cif = LR()
        model = cif.fit(X_train, y_train)
        result = cif.predict(X_test)
        # print()
        assert not (result[:, 0] == y_test).all()


if __name__ == '__main__':
    unittest.main(verbosity=1)
