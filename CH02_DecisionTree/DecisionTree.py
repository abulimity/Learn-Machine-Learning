import numpy as np
import pandas as pd
from math import log
import datetime
from functools import wraps


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
    labels = np.array(
        [
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
    )
    # dataLableMat = np.insert(dataSet, 4, labels, axis=1)
    dataSet = pd.DataFrame(dataSet, columns=["age", "work", "house", "credit"])
    labelSet = pd.Series(labels, name="labels")
    return dataSet, labelSet


def timing(func):
    wraps(func)

    def timiingDec(*args, **kwargs):
        timing = 0.0
        startTime = datetime.datetime.now()
        func(*args, **kwargs)
        endTime = datetime.datetime.now()
        timing = (endTime - startTime).seconds
        print("use time: %.2f" % timing)

    return timiingDec


def getEnt(labels):
    # classifyCnt = {}
    labelNum = len(labels)
    classifyCnt = labels.value_counts()
    # for i in labels:
    #     classifyCnt[i] = classifyCnt.get(i, 0) + 1
    ent = 0.0
    for key, cnt in classifyCnt.items():
        p = cnt / labelNum
        ent -= p * log(p, 2)
    return ent


def maxInfoGain(dataSet, labelSet):
    rowNum = dataSet.shape[0]
    ent = getEnt(labelSet)

    bestFeature = {"name": "", "infoGain": 0.0}
    for feature in dataSet.columns:
        # print("Now Feature: %s" % feature)
        colData = dataSet[feature]
        colValCnt = colData.value_counts().to_dict()
        conEnt = 0.0
        for key, val in colValCnt.items():
            p = val / rowNum
            # print("Now unique value: %s" % key)
            conRow = colData[colData == key]
            conLabel = labelSet.loc[list(conRow.index)]
            con_ent = p * getEnt(conLabel)
            # print(con_ent)
            conEnt += con_ent
        infoGain = ent - conEnt
        print("Feature %s ,infoGain %f" % (feature, infoGain))
        if infoGain > bestFeature["infoGain"]:
            bestFeature["infoGain"] = infoGain
            bestFeature["name"] = feature
    return bestFeature


def createTree(dataSet, sigma):
    labelDict = dataSet["label"].value_counts().to_dict()
    featureNum = len(dataSet.columns) - 1
    tree = {}
    if len(labelDict) == 1:
        tree = {"label": list(labelDict.keys())[0]}
        return list(labelDict.keys())[0]
    elif featureNum == 0:
        tree["label"] = sorted(labelDict.items(), key=lambda x: x[1], reverse=True)[0][
            0
        ]
        return sorted(labelDict.items(), key=lambda x: x[1], reverse=True)[0][0]
    bestFeature = maxInfoGain(dataSet.drop("label", axis=1), dataSet.label)
    if bestFeature["infoGain"] < sigma:
        tree["label"] = sorted(labelDict.items(), key=lambda x: x[1], reverse=True)[0][
            0
        ]
        return sorted(labelDict.items(), key=lambda x: x[1], reverse=True)[0][0]
    else:
        bestFeatName = bestFeature["name"]
        Tree = {bestFeatName: {}}
        for val in list(dataSet[bestFeatName].value_counts().index):
            subDataSet = dataSet[dataSet[bestFeatName] == val].drop(
                bestFeatName, axis=1
            )
            Tree[bestFeatName][val] = createTree(subDataSet, sigma)
    return Tree


if __name__ == "__main__":
    dataSet, labelSet = createDataSet()
    dataSet["label"] = labelSet
    sigma = 0.0
    print(dataSet)
    Tree = createTree(dataSet, sigma)
    print(Tree)
