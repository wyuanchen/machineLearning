# coding=utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    # distance=sqDistances**0.5
    sortedDistanceIndices=sqDistances.argsort()
    classCount={}
    for i in range(k):
        voteIlable=labels[sortedDistanceIndices[i]]
        classCount[voteIlable]=classCount.get(voteIlable,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def fileToMatrix(fileName):
    '''
    把文件数据集转化为矩阵
    :param fileName:
    :return:
    '''
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    '''
    归一化
    :param dataSet:
    :return:
    '''
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    range=maxVals-minVals;
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(range,(m,1))
    return normDataSet,range,minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=fileToMatrix("/media/yuan/Windows8_OS/machinelearninginaction/Ch02/datingTestSet2.txt")
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d"%(classifierResult,datingLabels[i])
        if(classifierResult!=datingLabels[i]):
            errorCount+=1
    print "------------------------------------------------------------"
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))


def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input("percent of time spent playing video games?"))
    ffMiles=float(raw_input("frequent flier miles earned per year?"))
    iceCream=float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=fileToMatrix("/media/yuan/Windows8_OS/machinelearninginaction/Ch02/datingTestSet2.txt")
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0(inArr,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult-1]



# fileName="/media/yuan/Windows8_OS/machinelearninginaction/Ch02/datingTestSet2.txt"
# datingDataMat,datingLabels=fileToMatrix(fileName)
# datingDataMat,range,minVals=autoNorm(datingDataMat)
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()


