from svmMLiA import *

dataMat,labelMat=loadDataSet('/media/yuan/Windows8_OS/machinelearninginaction/Ch06/testSet.txt',2)
print dataMat
print labelMat

# b,alphas=smoSimple(dataMat,labelMat,0.6,0.001,40,kTup=('rbf',1.3))
svmModel=getModel(dataMat,labelMat,0.6,40,kTup=('rbf',1.3))


