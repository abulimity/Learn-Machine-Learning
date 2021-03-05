import numpy as np
from math import log


def createDataSet():
    dataSet = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 2],
            [1, 0, 1, 2],
            [2, 0, 1, 2],
            [2, 0, 1, 1],
            [2, 1, 0, 1],
            [2, 1, 0, 2],
            [2, 0, 0, 0],
        ]
    )
    # columns = ["年龄", "有工作", "有自己的房子", "信贷情况"]  # 特征标签
    labels = [
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ]

    return dataSet, labels


def getEnt(labels):
    classifyCnt = {}
    labelNum = len(labels)
    for i in labels:
        classifyCnt[i] = classifyCnt.get(i, 0) + 1
    ent = 0.0
    for key, cnt in classifyCnt.items():
        p = cnt / labelNum
        ent -= p * log(p, 2)
    return ent


def getConditionEnt(dataSet, labels):
    pass


def infoGain(dataSet, feature):
    pass


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    a = np.vstack((dataSet, labels))
    print(dataSet)
    print(a)

    # ent = getEnt(labels)
    # print(ent)
