import os
from parsers import mainParser
from datetime import datetime
import time
import random
import numpy as np
import psutil
import gc

from dataPreparation.dataLoad import dataLoadFromFile
from dataPreparation.dataPreparation import DataPreparation
from dataProcessing.dataProcessing import mainDataProcessing
from dataPreparation.dataFrame import convertDataFrame
from dataDedimension.parseData import parseData
from dataDedimension.tsneReducer import tsne
from dataPreparation.dataCorrection import dataCorrection,dataCorrectionForMerging
from dataPreparation.dataListToFile import saveListToFile

from dataClustering.dataKMeans import kmeansClustering
from dataClustering.dataKernel import kernelKMeans
from cleanClusters.cleanClusters import cleanClusters
from cleanClusters.repeatClustering import repeatClustering
from dataPreparation.buildReClusterDataset import buildReClusterDataset
from dataPreparation.dataReadToList import readcsvToList
from dataEvaluation.buildLookupDict import buildLookupDict
from dataEvaluation.dataMetrics import f1_score

from siameseNetwork.siameseNetwork import siameseNetworkMain
from memoryManagement.memoryRelease import releaseList
from memoryManagement.memoryCheck import memoryCheck

from clustersMerging.clustersMerging import combineClustersCentroidsAsFeatures,mergeClusters

if __name__ == "__main__":
    print("Start of the program")
    now_start = datetime.now()

    input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode = mainParser()

    the_feature_extract_file = "./output/features_extraction.csv"
    imgs_dir = '../sample_data/unlabeled'
    if os.path.isfile(the_feature_extract_file):
        print ("Start: Reading the existing tsne features file")
        features = dataLoadFromFile(the_feature_extract_file)
    else:
        thisDataPre = DataPreparation(input_dir)
        img_names = list(thisDataPre.loadImages())

        print('Found {} images'.format(len(img_names)))
        img_names = random.sample(img_names,len(img_names))

        thisDataPre = None
        memoryCheck()

        features = mainDataProcessing(model_name,img_names,imgs_dir)

        write_original_dir = './output/features_extraction_org.csv'
        features = convertDataFrame(features,img_names,write_original_dir)

        gc.collect()

        features = parseData(features,feature_cols,unique_col)
        features = tsne(features, dims=int(reduce), write_to=extract_csv_dir)
    image_paths = features['ID'].tolist()
    features_df_temp = features[:]
    del features['ID']
    dataset = dataCorrection(features)

    gc.collect()

    """"Clustering Part"""""

    # """"K-means"""""
    # n_clusters = 11
    # clusters_kmeans = kmeansClustering(dataset,image_paths,n_clusters)
    # write_kmeans_dir = './output/kmeans_clusters.csv'
    # saveListToFile(clusters_kmeans,write_kmeans_dir)
    #
    # write_kmeans_dir = None, None
    # gc.collect()
    #
    # """"First Kernel K-means"""""
    # sigma = 10
    # clusters_kernel_kmeans = kernelKMeans(dataset,image_paths,n_clusters,sigma)
    # write_kernel_kmeans_dir = './output/kernel_kmeans_clusters.csv'
    # saveListToFile(clusters_kernel_kmeans,write_kernel_kmeans_dir)
    #
    # dataset = None
    # gc.collect()

    """Repeat Clustering"""
    iter = 8
    n_clusters = 49
    sigma = 1.3
    repeatClustering(features_df_temp,image_paths,iter,sigma,n_clusters)

    image_paths=sigma=n_clusters=None

    """Merging clusters"""
    features_extraction = dataCorrectionForMerging(features_df_temp)
    features,cluster_names = combineClustersCentroidsAsFeatures(iter,features_extraction)

    features_extraction = None

    sigma = 10
    n_clusters = int(len(features)/(iter*4))
    kernel_kmeans_merging_clusters_clusters = kernelKMeans(features,cluster_names,n_clusters,sigma)

    features,cluster_names = None, None

    write_kernel_kmeans_merging_clusters_clusters_dir = './output/kernel_kmeans_merging_clusters_clusters.csv'
    saveListToFile(kernel_kmeans_merging_clusters_clusters,write_kernel_kmeans_merging_clusters_clusters_dir)

    write_kernel_kmeans_merging_clusters_clusters_dir = None

    merged_clusters = mergeClusters(kernel_kmeans_merging_clusters_clusters)

    kernel_kmeans_merging_clusters_clusters = None

    write_merged_clusters_dir = './output/merged_clusters.csv'
    saveListToFile(merged_clusters,write_merged_clusters_dir)

    """Evaluate Precision of Merged Clusters"""
    write_lookup_dir = './output/class_lookup.csv'
    if os.path.isfile(write_lookup_dir):
        label_lookup = readcsvToList(write_lookup_dir)
    else:
        print ("Note: there needs a class lookup csv file, please delete all exisiting files in output dir and re-run")

    label_lookup = buildLookupDict(label_lookup)

    # print ("--------K-means Clusters Precision:-------")
    # _, _, _, precision, recall, score = f1_score(clusters_kmeans, label_lookup)
    # print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(len(clusters_kmeans), precision, recall, score))
    #
    # print ("--------Kernel K-means Clusters Precision:-------")
    # _, _, _, precision, recall, score = f1_score(clusters_kernel_kmeans, label_lookup)
    # print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(len(clusters_kernel_kmeans), precision, recall, score))

    print ("--------Merged Clusters Precision:-------")
    _, _, _, precision, recall, score = f1_score(merged_clusters, label_lookup)
    print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(len(merged_clusters), precision, recall, score))
