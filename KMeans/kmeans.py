'''
Created on Feb 15, 2018
The implementation of k-means
Reference: 
    1, Pattern Recognition
    2, machine learning in action
@author: hongzhucui
'''

from numpy import *
import random
import matplotlib.pyplot as plt

def load_data(filename):
    '''
    a helper function load data from text file, which is tab deliminated
    @param filenmae: the file name of the text file.
    @return dataset: list, which reprensents the datamatrix we get from the text file. 
    @raise 
    '''
    dataset = []
    with open(filename) as f: 
        for line in f:
            dataset.append([float(i) for i in line.strip().split('\t')])
    return dataset

def euclidian(x, y):
    '''
    function calculate the euclidian distance between two vectors
    @param x: a list vector
    @param y: a list vector
    @return: float, the distance between the two 
    @raise
    '''
    return linalg.norm(x-y)
def generate_random_centroids(dataset, k):
    '''
    given the dataset, we generate k random mean values. 
    the centroids are not from the dataset
    and they are should be within the range 
    @param dataset: a list representing the data matrix loaded from the file. 
    @param k: the paramter, 
    @return: a list consisting centroid 
    '''
    mat = matrix(dataset)
    y = mat.shape[1]
    max_values = mat.max(1)
    min_values = mat.min(1)
    centroids = []
    for i in range(k):
        centroids.append([random.uniform(float(min_values[j]), float(max_values[j])) for j in range(y)])
    return centroids

def kmeans(dataset, k, distance_measure=euclidian, create_centroids = generate_random_centroids):
    '''
    the main function of k-means:
    @param dataset: 
    @param k:
    @param distance_measure:
    @param create_centroid: 
    @return centroids: a list of centroids when it stops. 
    @return assignments: a list of the assignment for each data sample. the assignment could be 0, 1, 2,..., k
    
    '''
    rows, cols = shape(dataset)
    data_array = array(dataset) # use an matrix
    centroids = create_centroids(dataset, k)
    assignment_changed = True
    # return:
    assignments = [-1]*rows
    # loop until the assignment does not change !
    while assignment_changed:
        assignment_changed = False
        # assignment step:
        # assign the data point to the nearest centroid
       
        for i in range(rows):
            min_index = -1; min_dist = float(inf)
            for  j in range(k):
                dist = distance_measure(data_array[i,], centroids[j])
                if dist < min_dist: 
                    min_index = j; min_dist = dist

            if assignments[i] != min_index: 
                assignment_changed = True
            assignments[i] = min_index    
        
        # update step: 
        # update the centroid according based on the assignment. 
        for i in range(len(centroids)):
            
            cluster = data_array[array(assignments) == i]
            centroids[i] = mean(cluster, axis = 0).tolist()
        # print centroids
        
    return centroids, assignments

    
dataset = load_data('testSet.txt')
datamat = array(dataset) # use an array
centroids, assignments = kmeans(dataset, 4)
# centroids

plt.scatter(array(centroids)[:,0], array(centroids)[:,1], marker = '+', s = 200, c = 'c') 

markers = ['^', 'o', 'D', '*']
colors = ['b', 'g', 'y', 'r']
for i in range(4):
        cluster = datamat[array(assignments) == i]
        plt.scatter(cluster[:,0], cluster[:,1], marker = markers[i], color = colors[i])
plt.show()
    