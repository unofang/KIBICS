import os
import gc
import math
from tqdm import tqdm
import numpy as np

from parsers import mainParser

from dataProcessing.dataProcessing import mainDataProcessingNotqdm
from dataPreparation.dataFrame import convertDataFrameNoSave
from dataDedimension.parseData import parseData
from dataDedimension.tsneReducer import tsnePlain
from dataPreparation.dataCorrection import dataCorrection
from dataPreparation.dataLoad import dataFileToList

from memoryManagement.memoryRelease import releaseList
from memoryManagement.memoryCheck import memoryCheck

def calculateMeanPointOfCluster(cluster,features_extraction):
    _,_,_,_,_,model_name,feature_cols,unique_col,reduce,_ = mainParser()

    gc.collect()

    dataset = [[float(x[1]),float(x[2])] for x in features_extraction if x[0] in cluster]

    gc.collect()

    x = [p[0] for p in dataset]
    y = [p[1] for p in dataset]

    centroid = [sum(x) / len(dataset), sum(y) / len(dataset)]

    return centroid

def calculateDistanceBetweenClusters(point_a,point_b):
    distance = math.sqrt( ((point_a[0]-point_b[0])**2)+((point_a[1]-point_b[1])**2) )

    return distance

def clustersMerging(clusters_a,clusters_b,features_extraction):
    memoryCheck()
    dist_array = [[0] * len(clusters_b) for i in range(len(clusters_a))]
    for i in tqdm(range(len(clusters_a))):
        gc.collect()
        centroid_a = calculateMeanPointOfCluster(clusters_a[i],features_extraction)
        for j in range(len(clusters_b)):
            centroid_b = calculateMeanPointOfCluster(clusters_b[j],features_extraction)
            distance = calculateDistanceBetweenClusters(centroid_a,centroid_b)
            dist_array[i][j] = distance
            gc.collect()

    memoryCheck()

    return dist_array

def getClustersCentroids(clusters,features_extraction):
    for i in range(len(clusters)):
        yield calculateMeanPointOfCluster(clusters[i],features_extraction)

def combineClustersCentroidsAsFeatures(iter,features_extraction):
    merged_features = []
    merged_cluster_names = []
    for i in tqdm(range(iter)):
        the_cleaned_clusters_dir = './output/cleaned_clusters_'+ str(i) +'.csv'
        if os.path.isfile(the_cleaned_clusters_dir):
            clusters = dataFileToList(the_cleaned_clusters_dir)
        features = list(getClustersCentroids(clusters,features_extraction))
        merged_features = merged_features + features
        cluster_names = [str(i)+'_'+str(j) for j in range(len(features))]
        merged_cluster_names = merged_cluster_names +cluster_names

    return merged_features,merged_cluster_names

def mergeClusters(clusters_clusters):
    merged_clusters = []
    for i in tqdm(range(len(clusters_clusters))):
        the_new_cluster = []
        for j in range(len(clusters_clusters[i])):
            cleaned_clusters_index = int(clusters_clusters[i][j].split('_',1)[0])
            cluster_index = int(clusters_clusters[i][j].split('_',1)[1])
            the_cleaned_clusters_dir = './output/cleaned_clusters_'+ str(cleaned_clusters_index) +'.csv'
            if os.path.isfile(the_cleaned_clusters_dir):
                clusters = dataFileToList(the_cleaned_clusters_dir)
            the_cluster = clusters[cluster_index]
            the_new_cluster = the_new_cluster + the_cluster
        merged_clusters.append(the_new_cluster)

    return merged_clusters
