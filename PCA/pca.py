'''
Created on May 21, 2017

@author: hongzhucui
'''

from numpy import *

import string

def loadDataSet(filename, delim = '\t'):
    """
    load the data from text file
    @param filename
    @param delim, by default is TabError
    @return numpy matrix
    """
    fh = open (filename)
    # use two list comprehension to contruct the matrix
    
    strArr = [line.strip().split(delim) for line in fh.readlines()]
    dataArr = [map(float, line) for line in strArr]
    return mat(dataArr)

def pca(dataMat, pcNum = 999999999):
    """
    pca fuction, 
    @param dataMat, the design matrix
    @param pcNum, the # of the principle component we return
    @return lowDataMat the matrix based on the removed-mean matrix
    @return reconMat the reconstructed matrix 
    """
    meanVals = mean(dataMat)
    # calculate the matrix
    meanRemovedMat = dataMat - meanVals
    # calculate the covariance matrix
    covMat = cov(meanRemovedMat, rowvar = 0)
    # calculate the eigen value and eigen vectore use linalg Module
    eigenValues, eigenVectors = linalg.eig(mat(covMat))
    # sort the eigen value array
    eigenIndexes = argsort(eigenValues)
    # sorted array of eigen values.
    eigenIndexes = eigenIndexes[:-(pcNum+1):-1]
    # the first pcNum array. 
    reducedVectors = eigenVectors[:, eigenIndexes] 
    # reconstruct the matrix. 
    lowDataMat = meanRemovedMat*reducedVectors
    reconMat = (lowDataMat*reducedVectors.T) + meanVals # note this. 
    return lowDataMat, reconMat

if __name__ == "__main__":
    dataMat = loadDataSet('testSet.txt')
    lowMat, reconMat = pca(dataMat, 1)
    print shape(lowMat)
    
    