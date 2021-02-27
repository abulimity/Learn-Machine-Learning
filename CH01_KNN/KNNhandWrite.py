import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 查看当前工作目录
# print(os.getcwd())


def img2vector(fileDir: str):
    returnList = []
    with open(fileDir, "r") as f:
        fLines = f.readlines()
    for j in range(len(fLines)):
        line = fLines[j]
        for i in range(32):
            returnList.append(line[i])
    returnVector = pd.Series(returnList)
    # print(returnVector)
    return returnVector


def getDataMat():
    dataDir = r"data\CH01_KNN\trainingDigits"
    fileList = os.listdir(dataDir)
    m = len(fileList)
    hwLabels = []
    hwDataMat = np.zeros(shape=(m, 1024))
    for i in range(len(fileList)):
        fname = fileList[i]
        fileDir = dataDir + "\\" + fname
        hwLabels.append(int(fname.split("_")[0]))
        # print("xh: %s  labels: %s" % (fname.split("_")[-1].split(".")[0], fname.split("_")[0]))
        imgVector = img2vector(fileDir)
        hwDataMat[i] = imgVector
    return hwDataMat, hwLabels


def testClassify():
    dataDir = r"data\CH01_KNN\testDigits"
    fileList = os.listdir(dataDir)
    mTest = len(fileList)
    errorCnt = 0.0
    for fname in fileList:
        fileDir = dataDir + "\\" + fname
        testDataMat = img2vector(fileDir).reshape(1, -1)
        preResult = knn.predict(testDataMat)
        realResult = int(fname.split("_")[0])
        # print("预测结果是：%d------正确结果是：%d"%(preResult, realResult))
        if preResult != realResult:
            errorCnt += 1.0
    print("共测试样例%d个,预测错误%d个,错误率为%f%%" % (mTest, errorCnt, errorCnt / mTest * 100))


if __name__ == "__main__":
    # fileDir = r"..\data\CH01_KNN\trainingDigits\0_0.txt"
    # img2vector(fileDir)
    hwDataMat, hwLabels = getDataMat()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X=hwDataMat, y=hwLabels)
    testClassify()
