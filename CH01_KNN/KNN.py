import numpy as np
import pandas as pd
import operator


def readDatingData():
    head = ["flight", "game", "ice", "labels"]
    data = pd.read_csv(
        r"data\datingTestSet.txt", sep="\t", header=None, names=head, index_col=False
    )
    # data.head()
    datingDataMat = data.loc[:, ["flight", "game", "ice"]]
    datingLabels = data.loc[:, "labels"]
    return datingDataMat, datingLabels


def autoNorm(dataSet: pd.DataFrame, mode: str = "m") -> pd.DataFrame:
    # min-max方法
    # np.tile(data,(y,x))  np.tile(data,x) 平铺函数
    if mode == "m":
        diff = dataSet - np.tile(dataSet.min(), (dataSet.shape[0], 1))
        ranges = dataSet.max() - dataSet.min()
        normDataSet = diff / np.tile(ranges, (dataSet.shape[0], 1))
    # Z-score方法
    elif mode == "z":
        diff = dataSet - np.tile(dataSet.mean(), (dataSet.shape[0], 1))
        normDataSet = diff / np.tile(dataSet.std(), (dataSet.shape[0], 1))
    return normDataSet


def classify0(inX, dataSet, labels, k):
    # 数据行数
    dataSetSize = dataSet.shape[0]
    # 复制数据行数的inX 相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 相减后平方
    sqDiffMat = diffMat ** 2
    # 按行求和
    sqDistance = np.sum(sqDiffMat, axis=1)
    # 开方
    distance = sqDistance ** 0.5
    # 生成排序后的索引值
    sortedDistIndicies = distance.argsort()
    classCount = {}
    # 距离最近的前K个数据点分类出现频次统计
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # print(classCount)
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedClassCount[0][0]


def classifyTest(dataMat, labelsMat, hoRatio: float = 0.10):
    normDataSet = autoNorm(dataMat)
    testRowNum = int(normDataSet.shape[0] * hoRatio)
    testDataSet = normDataSet[testRowNum:].reset_index(drop=True)
    testLabels = labelsMat[testRowNum:].reset_index(drop=True)
    print(testRowNum)
    errorCount = 0.0
    for i in range(testRowNum):
        # print(i)
        classResult = classify0(
            normDataSet[i : i + 1],
            testDataSet,
            testLabels,
            4,
        )
        # print('真是结果：%s----预测结果：%s'%(datingLabels[i],classResult))
        if classResult != labelsMat[i]:
            errorCount += 1.0
    print("error rate is: %f" % (errorCount / float(testRowNum)))


def createDataSet():
    group = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ["爱情片", "爱情片", "爱情片", "动作片", "动作片", "动作片"]

    return group, labels


createDataSet()

group, labels = createDataSet()
# group
# classify0([18, 90], group, labels, 2)

if __name__ == "__main__":
    classify0([18, 90], group, labels, 2)
#     datingDataMat, datingLabels = readDatingData()
#     dataMat = datingDataMat[:20]
#     labelsMat = datingLabels[:20]
# classifyTest(dataMat, labelsMat)
# a = pd.Series([3,2,1])
# print(a.argsort())
