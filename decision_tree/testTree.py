# coding=utf-8
from trees import *
import treePlotter

# dataSet, labels = createDataSet()
# # print calShannonEnt(dataSet)
# # print chooseBestFeatureToSplit(dataSet)
# tree = createTree(dataSet, labels)
# print tree
#
#
# # treePlotter.createPlot()
#
#
# print classify(tree,labels,[1,0])
# print classify(tree,labels,[1,1])

dataSet, labels = fileToDataSet("/media/yuan/Windows8_OS/machinelearninginaction/Ch03/lenses.txt")
tree = createTree(dataSet, labels)
treePlotter.createPlot(tree)
