# coding=utf-8
import random

from numpy import array, zeros, ones
from numpy.ma import log


def loadDataSet():
    pass


def bagOfWordsToVecMN(vocabList, words):
    '''
    获取词袋模型
    :param vocabList:
    :param words:
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in words:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def createVocabList(documents):
    '''
    返回词条向量所含属性的list
    :param documents: 包含所有文档的文档列表，其中每一个文档都是一个包含词条的列表
    :return: vocabSet: 贝叶斯算法用到的词条向量所含属性的list
    '''
    vocabSet = set([])
    for document in documents:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def textParse(bigStr):
    '''
    分割字符串为一个词组列表
    :param bigStr:
    :return:
    '''
    import re
    listOfTokens = re.split(r'\W', bigStr)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def trainNB0(trainMatrix, trainClassMatrix):
    '''
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练集矩阵
    :param trainClassMatrix: 包含训练集的类别矩阵
    :return: p0Vect: 第0类的一个向量，其中每个维度的值表示在第0类的所有样本的所有词组中，该维度的词所占的比例, p1Vect: 意思和p0Vect类同, pAbusive: 第1类样本的数量在训练集所占的比例
    '''
    # 获取训练集有多少个样本
    numOfTrainDocs = len(trainMatrix)
    # 获取词条向量所包含的属性个数
    numOfWords = len(trainMatrix[0])
    pAbusive = sum(trainClassMatrix) / numOfTrainDocs
    p0Num = ones(numOfWords)
    p1Num = ones(numOfWords)
    # 分母，表示在第0类下训练集所包含的词的总数量
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numOfTrainDocs):
        if trainClassMatrix[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 这里用log是防止下溢
    p0Vect = log(p0Num / p0Denom)
    p1Vect = log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vecToClassify, p0Vec, p1Vec, pClass1):
    '''
    利用训练好的模型进行分类
    :param param:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1 = sum(vecToClassify * p0Vec) + log(pClass1)
    p0 = sum(vecToClassify * p1Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('/media/yuan/Windows8_OS/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('/media/yuan/Windows8_OS/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWordsToVecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWordsToVecMN(vocabList, docList[docIndex])
        result = classifyNB(array(wordVector), p0V, p1V, pSpam)
        if result != classList[docIndex]:
            errorCount += 1
    print 'the error count is: ',errorCount
    print 'the error rate is: ', float(float(errorCount) / len(testSet))
