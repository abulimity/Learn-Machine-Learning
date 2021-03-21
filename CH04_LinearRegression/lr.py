import numpy as np
import pandas as pd
import logging
from numpy.linalg import LinAlgError


class LR(object):
    def __init__(self):
        self.w = 0
        self.yHat = 0.0

    def fit(self, X, y):

        X = np.array(pd.DataFrame(X).to_numpy())
        X = np.mat(np.hstack([X, np.ones(shape=X.shape[0]).reshape(-1, 1)]))
        y = np.mat(np.array(pd.DataFrame(y).to_numpy()))
        self.w = np.zeros(shape=X.shape[1]+1)

        xTx = X.T * X
        xTy = X.T * y
        try:
            xTxI = xTx.I
        except LinAlgError as e:
            print("矩阵不可逆：%s" % e)
        self.w = xTx.I * xTy
        logging.info("got wMat:%s" % self.w)

    def predict(self, X):
        X = np.array(pd.DataFrame(X).to_numpy())
        X = np.mat(np.hstack([X, np.ones(shape=X.shape[0]).reshape(-1, 1)]))
        self.yHat = X * self.w
        return np.array(self.yHat)


if __name__ == '__mian__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # logging.info(os.getcwd())
