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
        write_original_dir = './output/features_extraction_org.csv'
        features_df = convertDataFrame(features,img_names,write_original_dir)
        df = parseData(features_df,feature_cols,unique_col)
        features_df = tsne(df, dims=int(reduce), write_to=extract_csv_dir)
    image_paths = features_df['ID'].tolist()
    features_df_temp = features_df[:]
    del features_df['ID']
    dataset = dataCorrection(features_df)


    """"Clustering Part"""""

    n_clusters = 50
    clusters_kmeans = kmeansClustering(dataset,image_paths,n_clusters)
    write_kmeans_dir = './output/kmeans_clusters.csv'
    saveListToFile(clusters_kmeans,write_kmeans_dir)
