# coding=utf-8
from math import log
import operator


def createDataSet():
    '''
    创建数据集
    :return:
    dataSet: 数据集
    labels: 属性标签列表
    '''
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calShannonEnt(dataSet):
    '''
    计算给定数据集的的香农熵
    :param dataSet:数据集
    :return: 返回香农熵
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, i, value):
    '''
    按照给定特征划分数据集
    :param dataSet:被划分的数据集
    :param axis:指通过第i列来划分
    :param value:
    :return:新的数据集，该数据集没有原来数据集上的第i列
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[i] == value:
            reducedFeatVec = featVec[:i]
            reducedFeatVec.extend(featVec[i + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet:原始数据集
    :return: bestFeature:表示所选择最优的属性的下标
    '''
    dataSetLen = len(dataSet)
    # 看还有多少有特征可以进行划分
    numFeatures = len(dataSet[0]) - 1
    baseEntroy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 依次检查每个特征
    for i in range(numFeatures):
        # 获取数据集在第i个列上的所有值(也就是第i个特征的所有取值)
        featureList = [row[i] for row in dataSet]
        # 用set来去重复
        uniqueVals = set(featureList)
        newEntroy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(dataSetLen)
            newEntroy += prob * calShannonEnt(subDataSet)
        # 计算用第i个属性来划分数据集得到的信息增益
        infoGain = baseEntroy - newEntroy
        # 更新最优的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    计算该样本中最多的类
    :param classList:
    :return: 该类的名字
    '''
    classCount = {}
    for eachClass in classList:
        classCount[eachClass] = classCount.get(eachClass, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    构造决策树
    :param dataSet: 数据集
    :param labels: 属性标签列表
    :return: tree: 决策树
    '''
    classList = [row[-1] for row in dataSet]
    if (classList.count(classList[0]) == len(classList)):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeatureIndex]
    tree = {bestFeatLabel: {}}
    # 复制下labels
    labels = labels[:]
    # 从属性标签列表中删除选中的属性
    del (labels[bestFeatureIndex])
    featValues = [row[bestFeatureIndex] for row in dataSet]
    uniqueFeatValue = set(featValues)
    for value in uniqueFeatValue:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeatureIndex, value)
        tree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
    return tree


def classify(tree, featLabels, testVec):
    '''
    利用决策树进行分类
    :param tree: 决策树
    :param featLabels: 包含所有属性的列表
    :param testVec: 测试的数据列表
    :return:
    '''
    firstStr = tree.keys()[0]
    secondDict = tree[firstStr]
    featIndex = featLabels.index(firstStr)
    keyValue = testVec[featIndex]
    result = secondDict[keyValue]
    # 如果该节点不是叶子节点
    if type(result).__name__ == 'dict':
        classLabel = classify(result, featLabels, testVec)
    else:
        # 该节点是叶子节点
        classLabel = result
    return classLabel


def storeTree(tree, fileName):
    '''
    存储树
    :param tree:
    :param fileName:
    '''
    import pickle
    fw = open(fileName, 'w')
    pickle.dump(tree, fw)
    fw.close()


def grabTree(fileName):
    '''
    从文件中取出数
    :param fileName:
    :return: 决策树
    '''
    import pickle
    fr = open(fileName)
    return pickle.load(fr)


def fileToDataSet(fileName):
    fr = open(fileName)
    dataSet = [row.strip().split('\t') for row in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet, labels
