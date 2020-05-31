import os
from parsers import mainParser
from datetime import datetime
import time
import random
import numpy as np

import keras
import keras.applications as kapp

# from clusterCombination.clusterCombination import clusterCombination
from dataPreparation.dataLoad import dataLoadFromFile
from dataPreparation.dataLoad import dataFileToList
# from dataEvaluation.dataMetrics import f1_score
from dataPreparation.dataReadToList import readcsvToList
# from dataEvaluation.buildLookupDict import buildLookupDict
from siameseNetwork.siameseNetwork import siameseNetworkMain
from dataPreparation.dataListToFile import saveListToFile

from clustersMerging.clustersMerging import clustersMerging,getClustersCentroids,combineClustersCentroidsAsFeatures,mergeClusters

from dataClustering.dataKernel import kernelKMeans

from dataEvaluation.buildLookupDict import buildLookupDict
from dataEvaluation.dataMetrics import f1_score

from dataClassification.dataTrain import trainModel
from dataClassification.dataPredict import predictTheImg,predictTheImgWithSimilarity

def buildModel(model_name):
    """ Create a pretrained model without the final classification layer. """
    if model_name == "resnet50":
        model = kapp.resnet50.ResNet50(weights="imagenet", include_top=False)
        return model, kapp.resnet50.preprocess_input
    elif model_name == "vgg16":
        model = kapp.vgg16.VGG16(weights="imagenet", include_top=False)
        return model, kapp.vgg16.preprocess_input
    else:
        raise Exception("Unsupported model error")

if __name__ == "__main__":
    input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode = mainParser()

    the_merged_clusters_file = "./output/merged_clusters.csv"
    if os.path.isfile(the_merged_clusters_file):
        clusters = dataFileToList(the_merged_clusters_file)

    iter = 8
    the_abnormal_cluster_imgs_file = "./output/abnormal_cluster_imgs_"+str(iter)+".csv"
    if os.path.isfile(the_abnormal_cluster_imgs_file):
        abnormal_cluster_imgs = dataFileToList(the_abnormal_cluster_imgs_file)

    imgs_dir = '../sample_data/unlabeled'
    trainedModel,predictingClasses = trainModel(clusters,imgs_dir)

    clusters = predictTheImg(trainedModel,predictingClasses,imgs_dir,abnormal_cluster_imgs,clusters,model_name)

# if __name__ == "__main__":
#     input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode = mainParser()
#
#     the_clusters_a_file = "./output/re_kernel_kmeans_clusters.csv"
#     if os.path.isfile(the_clusters_a_file):
#         clusters_a = dataFileToList(the_clusters_a_file)
#
#     the_clusters_b_file = "./output/re_kernel_kmeans_clusters_a.csv"
#     if os.path.isfile(the_clusters_b_file):
#         clusters_b = dataFileToList(the_clusters_b_file)
#
#     the_features_tsne_file = "./output/features_extraction.csv"
#     if os.path.isfile(the_features_tsne_file):
#         features_extraction = dataFileToList(the_features_tsne_file)

    # imgs_dir = '../sample_data/unlabeled'

    # dist_array = clustersMerging(clusters_a,clusters_b,features_extraction)
    #
    # write_dist_array_dir = './output/dist_array.csv'
    # saveListToFile(dist_array,write_dist_array_dir)

    # feature_a = list(getClustersCentroids(clusters_a,features_extraction))
    # cluster_names_a = ['A_'+str(i) for i in range(len(feature_a))]
    #
    # feature_b = list(getClustersCentroids(clusters_b,features_extraction))
    # cluster_names_b = ['B_'+str(i) for i in range(len(feature_b))]
    #
    # features = feature_a + feature_b
    # cluster_names = cluster_names_a + cluster_names_b
    #
    # write_features_extraction_clusters_dir = './output/features_extraction_clusters.csv'
    # saveListToFile(features,write_features_extraction_clusters_dir)
    #
    # sigma = 2
    # n_clusters = int(len(features)/2)
    # kernel_kmeans_on_two_clusters = kmeansClustering(features,cluster_names,n_clusters)
    # write_kernel_kmeans_on_two_clusters_dir = './output/kernel_kmeans_on_two_clusters.csv'
    # saveListToFile(kernel_kmeans_on_two_clusters,write_kernel_kmeans_on_two_clusters_dir)

    # iter = 10
    # features,cluster_names = combineClustersCentroidsAsFeatures(iter,features_extraction)
    #
    # sigma = 10
    # n_clusters = int(len(features)/(iter*4))
    # kernel_kmeans_merging_clusters_clusters = kernelKMeans(features,cluster_names,n_clusters,sigma)
    #
    # write_kernel_kmeans_merging_clusters_clusters_dir = './output/kernel_kmeans_merging_clusters_clusters.csv'
    # saveListToFile(kernel_kmeans_merging_clusters_clusters,write_kernel_kmeans_merging_clusters_clusters_dir)
    #
    # merged_clusters = mergeClusters(kernel_kmeans_merging_clusters_clusters)
    #
    # write_merged_clusters_dir = './output/merged_clusters.csv'
    # saveListToFile(merged_clusters,write_merged_clusters_dir)
    #
    # write_lookup_dir = './output/class_lookup.csv'
    # if os.path.isfile(write_lookup_dir):
    #     label_lookup = readcsvToList(write_lookup_dir)
    # else:
    #     print ("Note: there needs a class lookup csv file, please delete all exisiting files in output dir and re-run")
    #
    # label_lookup = buildLookupDict(label_lookup)
    #
    # print ("--------Merged Clusters Precision:-------")
    # _, _, _, precision, recall, score = f1_score(merged_clusters, label_lookup)
    # print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(len(merged_clusters), precision, recall, score))


# if __name__ == "__main__":
#     the_clusters_a_file = "./output/re_kernel_kmeans_clusters.csv"
#     if os.path.isfile(the_clusters_a_file):
#         print ("Start: Reading the existing new kernel kmeans clusters file")
#         clusters_a = dataFileToList(the_clusters_a_file)
#
#     the_clusters_b_file = "./output/re_kernel_kmeans_clusters_a.csv"
#     if os.path.isfile(the_clusters_b_file):
#         print ("Start: Reading the existing new kernel kmeans clusters file")
#         clusters_b = dataFileToList(the_clusters_b_file)
#
#     imgs_dir = '../sample_data/unlabeled'
#     siameseNetworkMain(imgs_dir,clusters_a,clusters_b)

# if __name__ == "__main__":
#     the_clusters_file = "./output/re_kernel_kmeans_clusters.csv"
#     if os.path.isfile(the_clusters_file):
#         print ("Start: Reading the existing new kernel kmeans clusters file")
#         clusters = dataFileToList(the_clusters_file)
#
#     the_top_x_array_file = "./output/top_x_array.csv"
#     if os.path.isfile(the_top_x_array_file):
#         print ("Start: Reading the existing top_x_array file")
#         top_x_array = dataFileToList(the_top_x_array_file)
#
#     imgs_dir = '../sample_data/unlabeled'
#     model = siameseNetworkTrain(clusters,imgs_dir)
#
#     the_to_cluster_images_file = "./output/to_cluster_images.csv"
#     if os.path.isfile(the_to_cluster_images_file):
#         print ("Start: Reading the to-be-clustered images")
#         input_images_cl = dataFileToList(the_to_cluster_images_file)
#         input_images = [input[0] for input in input_images_cl]
#
#     predict = siameseNetworkPredict(model,imgs_dir,input_images,clusters,top_x_array)
#
#     write_test_predict_dir = './output/test_predict.csv'
#     saveListToFile(predict,write_test_predict_dir)

    # write_test_len_array_dir = './output/test_len_array.csv'
    # saveListToFile(len_array,write_test_len_array_dir)
# if __name__ == "__main__":
#     print("Start of the program")
#     now_start = datetime.now()
#     input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode = mainParser()
#
#     the_clusters_file = "./output/new_kernel_kmeans_clusters.csv"
#     if os.path.isfile(the_clusters_file):
#         print ("Start: Reading the existing new kernel kmeans clusters file")
#         clusters = dataFileToList(the_clusters_file)
#
#     model, preprocess_fn = buildModel(model_name)
#     imgs_dir = '../sample_data/unlabeled'
#     combined_clusters = clusterCombination(clusters,imgs_dir,model,preprocess_fn)
#
#     combined_clusters = clusterCombination(combined_clusters,imgs_dir,model,preprocess_fn)
#
#     print (combined_clusters)
#
#     write_lookup_dir = './output/class_lookup.csv'
#     if os.path.isfile(write_lookup_dir):
#         label_lookup = readcsvToList(write_lookup_dir)
#     else:
#         print ("Note: there needs a class lookup csv file, please delete all exisiting files in output dir and re-run")
#
#     label_lookup = buildLookupDict(label_lookup)
#
#     print ("--------Combined Clusters:-------")
#     _, _, _, precision, recall, score = f1_score(combined_clusters, label_lookup)
#     print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(len(combined_clusters), precision, recall, score))
