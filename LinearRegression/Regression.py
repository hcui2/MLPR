'''
Created on Apr 27, 2017

@author: hongzhucui
'''

from numpy import *


def loadDataSet(fileName):
    """
    load the data from text file 
    @param fiele: the file name
    @return xMat, labelMat, both are the list of list (float). 
    """
    numFeatures = len(open(fileName).readline().split('\t')) - 1 
    xMat = []; labelMat = []
    fh = open (fileName)
    for line in fh.readlines():
        lineArray = line.strip().split('\t')
        sampleArr = []
        for i in range(numFeatures):
            sampleArr.append(float(lineArray[i]))
        xMat.append(sampleArr)
        labelMat.append(float(lineArray[-1]))
    return xMat, labelMat

def standardRegression(xArr, labelArr):
    """
    an implemenetation of standard regression
    @param xMat: the matrix represented as list
    @param labelMat: the label matrix represened as list
    @return: the weigtht matrix.   
    """
    xMat = mat(xArr); yMat = mat(labelArr).T 
    xTx = xMat.T * xMat 

    # calculate the inverse
    if linalg.det(xTx) == 0:
        print "the matrix does not have an inverse"
        return
    # invX = linalg.inv(xTx)
    # w = invX*xMat.T*yMat
    w =  xTx.I *(xMat.T*yMat)
    return w 