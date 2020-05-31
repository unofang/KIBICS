import os

from dataPreparation.buildReClusterDataset import buildReClusterDataset
from dataPreparation.dataCorrection import dataCorrection,dataCorrectionForMerging

from dataClustering.dataKernel import kernelKMeans
from cleanClusters.cleanClusters import cleanClusters

from dataPreparation.dataListToFile import saveListToFile

def repeatClustering(features,image_paths,iter,sigma,n_clusters):
    for i in range(iter):
        dataset = buildReClusterDataset(features,image_paths)
        del dataset['ID']
        dataset = dataCorrection(dataset)

        clusters_kernel_kmeans = kernelKMeans(dataset,image_paths,n_clusters,sigma)
        abnormal_cluster_imgs, cleaned_clusters = cleanClusters(clusters_kernel_kmeans)

        write_cleaned_clusters_dir = './output/cleaned_clusters_'+ str(i) +'.csv'
        saveListToFile(cleaned_clusters,write_cleaned_clusters_dir)

        write_abnormal_cluster_imgs_dir = './output/abnormal_cluster_imgs_'+ str(i) +'.csv'
        saveListToFile(abnormal_cluster_imgs,write_abnormal_cluster_imgs_dir)

        image_paths = abnormal_cluster_imgs
