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
from dataPreparation.buildReClusterDataset import buildReClusterDataset

from siameseNetwork.siameseNetwork import siameseNetworkMain
from memoryManagement.memoryRelease import releaseList
from memoryManagement.memoryCheck import memoryCheck

if __name__ == "__main__":
    print("Start of the program")
    now_start = datetime.now()

    input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode = mainParser()

    the_feature_extract_file = "./output/features_extraction.csv"
    imgs_dir = '../sample_data/unlabeled'
    if os.path.isfile(the_feature_extract_file):
        print ("Start: Reading the existing tsne features file")
        features_df = dataLoadFromFile(the_feature_extract_file)
    else:
        thisDataPre = DataPreparation(input_dir)
        img_names = list(thisDataPre.loadImages())

        print('Found {} images'.format(len(img_names)))
        img_names = random.sample(img_names,len(img_names))

        memoryCheck()

        features = mainDataProcessing(model_name,img_names,imgs_dir)

        write_original_dir = './output/features_extraction_org.csv'
        features_df = convertDataFrame(features,img_names,write_original_dir)

        gc.collect()

        df = parseData(features_df,feature_cols,unique_col)
        features_df = tsne(df, dims=int(reduce), write_to=extract_csv_dir)
    image_paths = features_df['ID'].tolist()
    features_df_temp = features_df[:]
    del features_df['ID']
    dataset = dataCorrection(features_df)

    gc.collect()

    """"Clustering Part"""""

    n_clusters = 50
    clusters_kmeans = kmeansClustering(dataset,image_paths,n_clusters)
    write_kmeans_dir = './output/kmeans_clusters.csv'
    saveListToFile(clusters_kmeans,write_kmeans_dir)

    sigma = 1.2
    clusters_kernel_kmeans = kernelKMeans(dataset,image_paths,n_clusters,sigma)
    write_kernel_kmeans_dir = './output/kernel_kmeans_clusters.csv'
    saveListToFile(clusters_kernel_kmeans,write_kernel_kmeans_dir)

    print ("Start: Clean Kernel-K-means clusters")
    re_cluster_imgs, re_kernel_kmeans_clusters = cleanClusters(clusters_kernel_kmeans[:])
    for i in range(len(re_kernel_kmeans_clusters)):
        print("Cluster {} containing {} objects".format(i,len(re_kernel_kmeans_clusters[i])))

    write_to_cluster_images_dir = './output/to_cluster_images.csv'
    write_to_cluster_images = [ [x] for x in re_cluster_imgs]
    saveListToFile(write_to_cluster_images,write_to_cluster_images_dir)

    write_re_kernel_kmeans_dir = './output/re_kernel_kmeans_clusters.csv'
    saveListToFile(re_kernel_kmeans_clusters,write_re_kernel_kmeans_dir)

    re_cluster_features_df = buildReClusterDataset(features_df_temp,re_cluster_imgs)
    re_cluster_image_paths = re_cluster_features_df['ID'].tolist()
    del re_cluster_features_df['ID']
    re_cluster_dataset = dataCorrection(re_cluster_features_df)

    the_sigma = 1
    second_clusters_kernel_kmeans = kernelKMeans(re_cluster_dataset,re_cluster_image_paths,n_clusters,the_sigma)
    write_second_kernel_kmeans_dir = './output/second_kernel_kmeans_clusters.csv'
    saveListToFile(second_clusters_kernel_kmeans,write_second_kernel_kmeans_dir)

    print ("Start: Second Clean Kernel-K-means clusters")
    re_cluster_imgs_a, re_kernel_kmeans_clusters_a = cleanClusters(second_clusters_kernel_kmeans[:])
    for i in range(len(re_kernel_kmeans_clusters_a)):
        print("Cluster {} containing {} objects".format(i,len(re_kernel_kmeans_clusters_a[i])))

    write_to_cluster_images_a_dir = './output/to_cluster_images_a.csv'
    write_to_cluster_images_a = [ [x] for x in re_cluster_imgs_a]
    saveListToFile(write_to_cluster_images_a,write_to_cluster_images_a_dir)

    write_re_kernel_kmeans_a_dir = './output/re_kernel_kmeans_clusters_a.csv'
    saveListToFile(re_kernel_kmeans_clusters_a,write_re_kernel_kmeans_a_dir)

    memoryCheck()

    releaseList(clusters_kmeans)
    releaseList(clusters_kernel_kmeans)
    releaseList(clusters_kernel_kmeans)
    gc.collect()

    memoryCheck()

    imgs_dir = '../sample_data/unlabeled'
    siameseNetworkMain(imgs_dir,re_kernel_kmeans_clusters,re_kernel_kmeans_clusters_a)

    """"First Cleaned Kernel K-means Clusters"""""
    print ("Start: Clean Kernel-K-means clusters")
    to_cluster_imgs_for_second, kernel_kmeans_clusters_for_second = cleanClusters(clusters_kernel_kmeans[:])
    for i in range(len(kernel_kmeans_clusters_for_second)):
        print("Cluster {} containing {} objects".format(i,len(kernel_kmeans_clusters_for_second[i])))

    clusters_kernel_kmeans,write_kernel_kmeans_dir = None, None
    gc.collect()

    """"First To-be-re-clustered images"""""
    write_to_cluster_images_for_second_dir = './output/to_cluster_images_for_second.csv'
    to_cluster_images_for_second = [ [x] for x in to_cluster_imgs_for_second]
    saveListToFile(to_cluster_images_for_second,write_to_cluster_images_for_second_dir)

    to_cluster_images_for_second,write_to_cluster_images_for_second_dir = None, None
    gc.collect()

    """"First Cleaned Kernel K-means Clusters"""""
    write_kernel_kmeans_clusters_for_second_dir = './output/kernel_kmeans_clusters_for_second.csv'
    saveListToFile(kernel_kmeans_clusters_for_second,write_kernel_kmeans_clusters_for_second_dir)

    write_kernel_kmeans_clusters_for_second_dir = None

    """"Preparation for the second kernel k-means clustering"""""
    second_cluster_features_df = buildReClusterDataset(features_df_temp,to_cluster_imgs_for_second)
    second_cluster_image_paths = second_cluster_features_df['ID'].tolist()
    del second_cluster_features_df['ID']
    second_cluster_dataset = dataCorrection(second_cluster_features_df)

    to_cluster_imgs_for_second,second_cluster_features_df = None, None

    """"Second Kernel K-means Clusters"""""
    the_sigma = 1
    second_clusters_kernel_kmeans = kernelKMeans(second_cluster_dataset,second_cluster_image_paths,n_clusters,the_sigma)

    second_cluster_dataset,second_cluster_image_paths = None, None

    """"Second Cleaned Kernel K-means Clusters"""""
    print ("Start: Clean Kernel-K-means clusters secondly")
    to_cluster_imgs_for_third, kernel_kmeans_clusters_for_third = cleanClusters(second_clusters_kernel_kmeans[:])
    for i in range(len(kernel_kmeans_clusters_for_third)):
        print("Cluster {} containing {} objects".format(i,len(kernel_kmeans_clusters_for_third[i])))

    second_clusters_kernel_kmeans = None

    """"Second To-be-re-clustered images"""""
    write_to_cluster_images_for_third_dir = './output/to_cluster_images_for_third.csv'
    to_cluster_images_for_third = [ [x] for x in to_cluster_imgs_for_third]
    saveListToFile(to_cluster_images_for_third,write_to_cluster_images_for_third_dir)

    write_to_cluster_images_for_third_dir = None

    """"Second Cleaned Kernel K-means Clusters"""""
    write_kernel_kmeans_clusters_for_third_dir = './output/kernel_kmeans_clusters_for_third.csv'
    saveListToFile(kernel_kmeans_clusters_for_third,write_kernel_kmeans_clusters_for_third_dir)

    write_kernel_kmeans_clusters_for_third_dir = None

    """"Preparation for the third kernel k-means clustering"""""
    third_cluster_features_df = buildReClusterDataset(features_df_temp,to_cluster_imgs_for_third)
    third_cluster_image_paths = third_cluster_features_df['ID'].tolist()
    del third_cluster_features_df['ID']
    third_cluster_dataset = dataCorrection(third_cluster_features_df)

    to_cluster_imgs_for_third,third_cluster_features_df = None, None

    """"Third Kernel K-means Clusters"""""
    the_sigma = 1
    third_clusters_kernel_kmeans = kernelKMeans(third_cluster_dataset,third_cluster_image_paths,n_clusters,the_sigma)

    third_cluster_dataset,third_cluster_image_paths = None, None

    """"Third Cleaned Kernel K-means Clusters"""""
    print ("Start: Clean Kernel-K-means clusters thirdly")
    to_cluster_imgs_for_final, kernel_kmeans_clusters_for_final = cleanClusters(third_clusters_kernel_kmeans[:])
    for i in range(len(kernel_kmeans_clusters_for_final)):
        print("Cluster {} containing {} objects".format(i,len(kernel_kmeans_clusters_for_final[i])))

    third_clusters_kernel_kmeans = None

    """"Third To-be-re-clustered images"""""
    write_to_cluster_images_for_final_dir = './output/to_cluster_images_for_final.csv'
    to_cluster_images_for_final = [ [x] for x in to_cluster_imgs_for_final]
    saveListToFile(to_cluster_images_for_final,write_to_cluster_images_for_final_dir)

    to_cluster_imgs_for_final,write_to_cluster_images_for_final_dir = None, None

    """"Final Cleaned Kernel K-means Clusters"""""
    write_kernel_kmeans_clusters_for_final_dir = './output/kernel_kmeans_clusters_for_final.csv'
    saveListToFile(kernel_kmeans_clusters_for_final,write_kernel_kmeans_clusters_for_final_dir)

    write_final_kernel_kmeans_clusters_dir = None

    """Merging Clusters"""
    features_extraction = dataCorrectionForMerging(features_df_temp)

    feature_1 = list(getClustersCentroids(kernel_kmeans_clusters_for_second,features_extraction))
    cluster_names_1 = ['A_'+str(i) for i in range(len(feature_1))]

    feature_2 = list(getClustersCentroids(kernel_kmeans_clusters_for_third,features_extraction))
    cluster_names_2 = ['B_'+str(i) for i in range(len(feature_2))]

    feature_3 = list(getClustersCentroids(kernel_kmeans_clusters_for_final,features_extraction))
    cluster_names_3 = ['C_'+str(i) for i in range(len(feature_3))]

    features_extraction = None
    features_df_temp = None

    features = feature_1 + feature_2 + feature_3
    cluster_names = cluster_names_1 + cluster_names_2 + cluster_names_3

    write_for_merging_clusters_features_dir = './output/for_merging_clusters_features.csv'
    saveListToFile(features,write_for_merging_clusters_features_dir)

    sigma = 2
    n_clusters = int(len(features)/6)
    kernel_kmeans_merging_clusters_clusters = kmeansClustering(features,cluster_names,n_clusters)
    write_kernel_kmeans_merging_clusters_clusters_dir = './output/kernel_kmeans_merging_clusters_clusters.csv'
    saveListToFile(kernel_kmeans_merging_clusters_clusters,write_kernel_kmeans_merging_clusters_clusters_dir)

    features,cluster_names = None, None
    write_kernel_kmeans_merging_clusters_clusters_dir = None
