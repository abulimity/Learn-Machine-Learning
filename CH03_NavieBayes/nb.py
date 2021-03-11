import numpy as np
import pandas as pd
import logging


class NB(object):
    def __init__(self):
        self._py = None
        self._pyx = None
        self._yCnt = None
        self._yValCnt = None

    def fit(self, X, y):
        self._yCnt = np.unique(y)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        self._yValCnt = y.value_counts()
        self._py = self._yValCnt / y.shape[0]

        self._pyx = dict()
        for x in X.columns:
            for i in self._yCnt:
                x_y = X[(y == i).values][x].value_counts()
                for j in x_y.index:
                    self._pyx[(x, j, i)] = x_y[j] / self._yValCnt[i]
        return self._pyx

    def predict(self, X):
        result = []
        for i in self._yCnt:
            py = self._py[i]
            pyx = 1
            for idx, val in enumerate(X):
                pyx *= self._pyx[(idx, val, i)]
            result.append(pyx * py)
        return self._yCnt[np.argmax(result)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # x = pd.DataFrame(data=[[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']]).T
    # y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    # nb = NB()
    # model = nb.fit(x, y)
    # logger.info('after train, we got model: %s' % model)

    # x_test = [2,'S']
    # result = nb.predict(x_test)
    # logger.info('predict result: %s' % result)
