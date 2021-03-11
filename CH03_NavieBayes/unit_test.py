from nb import NB
import unittest
import numpy as np
import pandas as pd


class TestNB(unittest.TestCase):
    def test_log(self):
        x = pd.DataFrame(data=[[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']]).T
        y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
        cif = NB()
        model = cif.fit(x, y)
        x_test = [2, 'S']
        result = cif.predict(x_test)
        assert result == -1


if __name__ == '__main__':
    unittest.main(verbosity=1)
