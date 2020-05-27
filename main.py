import os
from parsers import mainParser
from datetime import datetime
import time
import random
import numpy as np
import psutil

from dataPreparation.dataLoad import dataLoadFromFile
from dataPreparation.dataPreparation import DataPreparation
from dataProcessing.dataProcessing import mainDataProcessing
from dataPreparation.dataFrame import convertDataFrame
from dataDedimension.parseData import parseData
from dataDedimension.tsneReducer import tsne
from dataPreparation.dataCorrection import dataCorrection
from dataPreparation.dataListToFile import saveListToFile

from dataClustering.dataKMeans import kmeansClustering
from dataClustering.dataKernel import kernelKMeans
from cleanClusters.cleanClusters import cleanClusters
from dataPreparation.buildReClusterDataset import buildReClusterDataset

from siameseNetwork.siameseNetwork import siameseNetworkMain
from memoryManagement.memoryRelease import releaseList,clearAllValuesExceptSelection

if __name__ == "__main__":
    print("Start of the program")
    now_start = datetime.now()

    input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode = mainParser()

    the_feature_extract_file = "./output/features_extraction.csv"
    if os.path.isfile(the_feature_extract_file):
        print ("Start: Reading the existing tsne features file")
        features_df = dataLoadFromFile(the_feature_extract_file)
    else:
        thisDataPre = DataPreparation(input_dir)
        train_images_with_filenames,class_lookup_df = thisDataPre.loadImages()
        print('Found {} images'.format(len(train_images_with_filenames)))
        train_images_with_filenames = random.sample(train_images_with_filenames,len(train_images_with_filenames))

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)  # in bytes

        features,img_names = mainDataProcessing(model_name, train_images_with_filenames)

        releaseList(train_images_with_filenames)

        write_original_dir = './output/features_extraction_org.csv'
        features_df = convertDataFrame(features,img_names,write_original_dir)
        df = parseData(features_df,feature_cols,unique_col)
        features_df = tsne(df, dims=int(reduce), write_to=extract_csv_dir)
    image_paths = features_df['ID'].tolist()
    features_df_temp = features_df[:]
    del features_df['ID']
    dataset = dataCorrection(features_df)

    the_feature_extract_file_org = "./output/features_extraction_org.csv"
    if os.path.isfile(the_feature_extract_file_org):
        print ("Start: Reading the existing original features file")
        features_df_org = dataLoadFromFile(the_feature_extract_file_org)
        del features_df_org['ID']
        dataset_org = dataCorrection(features_df_org)
    else:
        print ("Warn: You might need to delete all files in output folder and re-run")

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

    clearAllValuesExceptSelection(['re_kernel_kmeans_clusters','re_kernel_kmeans_clusters_a'])

    imgs_dir = '../sample_data/unlabeled'
    siameseNetworkMain(imgs_dir,re_kernel_kmeans_clusters,re_kernel_kmeans_clusters_a)
