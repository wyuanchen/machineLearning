# coding=utf-8
import random
from numpy import *


def loadDataSet(fileName, n):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        temp = []
        for i in range(n):
            temp.append(float(lineArr[i]))
        dataMat.append(temp)
        # dataMat.append(lineArr[:-1])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataSet, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    SMO简化算法
    :param dataMat: 数据集
    :param classLabels: 分类标签(1或者-1)
    :param C:
    :param toler:尽量小就行
    :param maxIter: (最大迭代次数)
    :return:
    '''
    dataMatrix = mat(dataSet)
    labelMatrix = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    K = calK(dataMatrix, m, kTup)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # fXi = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # Ei = fXi - float(labelMatrix[i])
            Ei = calEk(alphas, labelMatrix, dataMatrix, K, b, i)
            if (labelMatrix[i] * Ei < -toler) and (alphas[i] < C) or (labelMatrix[i] * Ei > toler) and (alphas[i] > 0):
                j = selectJrand(i, m)
                # fXj = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                # Ej = fXj - float(labelMatrix[j])
                Ej = calEk(alphas, labelMatrix, dataMatrix, K, b, j)
                alphaIOld = alphas[i].copy()
                alphaJOld = alphas[j].copy()
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if (L == H):
                    # print "L==H"
                    continue
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                # if eta >= 0:
                #     print "eta>=0"
                #     continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJOld) < 0.00001):
                    # print "j not moving enough"
                    continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJOld - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIOld) * K[i,i] - \
                     labelMatrix[j] * (alphas[j] - alphaJOld) * K[i,j]
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIOld) * K[i,j] - \
                     labelMatrix[j] * (alphas[j] - alphaJOld) * K[j,j]
                if (0 < alphas[i]) and (alphas[i] < C):
                    b = b1;
                elif (0 < alphas[j]) and (alphas[j] < C):
                    b = b2;
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                # print "iter: %d i:%d, paris changed %d " % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        # print "iteration number: %d" % iter
    return b, alphas


def kernelTrans(X, A, kTup):
    '''
    返回K矩阵的某一列
    :param X:
    :param A:
    :param kTup: 核函数的参数
    :return:
    '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if (kTup[0] == 'lin'):
        K = X * A.T
    elif (kTup[0] == 'rbf'):
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('The kernel is not recognized')
    return K


def calK(dataMatrix, m, kTup):
    '''
    计算K矩阵
    :param dataMatrix: 数据集矩阵
    :param m: 样本的数量
    :param kTup: 核函数的参数
    :return: K矩阵
    '''
    K = mat(zeros((m, m)))
    # 每一次循环都会算好K矩阵的第i列的所有值
    for i in range(m):
        K[:, i] = kernelTrans(dataMatrix, dataMatrix[i, :], kTup)
    return K


def calEk(alphas, labelMatrix, dataMatrix, K, b, k):
    fXk = float(multiply(alphas, labelMatrix).T * K[:, k] + b)
    Ek = fXk - float(labelMatrix[k])
    return Ek

def getModel(dataSet,labelMat,C,maxIter,kTup=('lin',0)):
    b,alphas=smoSimple(dataSet,labelMat,0.6,0.001,40,kTup)
    dataMatrix=mat(dataSet)
    labelMatrix=mat(labelMat).transpose()
    m,n=shape(dataMatrix)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMatrix[i],dataMatrix[i,:].T)
    return svmModel(w,b,alphas)



class svmModel:
    def __init__(self,w,b,alphas):
        self.w=w
        self.b=b
        self.alphas=alphas
        w=mat(w)

    def calF(self,lineAttr):
        lineAttrMat=mat(lineAttr).T
        f=self.w.T*lineAttrMat+self.b
        return f




def imgToDataLine(fileName):
    fr=open(fileName)
    dataVect=zeros((1,1024))
    label=int(fileName.split('_')[0])
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            dataVect[0,32*i+j]=int(line[j])
    return dataVect,label
