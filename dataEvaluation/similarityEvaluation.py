import os

from dataPreparation.dataReadToList import readcsvToList

def columnExtraction(matrix, i, j):
    if j == None:
        return [row[i] for row in matrix]
    else:
        return [row[i:j] for row in matrix]

def labelExtraction(str):
    first = str.split(' ',1)[0]
    second = first.split('___',1)[1]
    return second

def similarityEvaluation():
    write_to_cluster_images_dir = './output/to_cluster_images.csv'
    if os.path.isfile(write_to_cluster_images_dir):
        to_cluster_images = readcsvToList(write_to_cluster_images_dir)
        the_wait_to_labels = []
        for img_name in to_cluster_images:
            the_wait_to_labels.append(labelExtraction(img_name[0]))

    write_classification_proba_dir = './output/classification_proba.csv'
    if os.path.isfile(write_classification_proba_dir):
        classification_proba = readcsvToList(write_classification_proba_dir)
        classification_proba = columnExtraction(classification_proba,(len(classification_proba[0])-15),(len(classification_proba[0])-1))

    write_re_kernel_kmeans_clusters_dir = './output/re_kernel_kmeans_clusters.csv'
    if os.path.isfile(write_re_kernel_kmeans_clusters_dir):
        re_kernel_kmeans_clusters = readcsvToList(write_re_kernel_kmeans_clusters_dir)
        re_kernel_kmeans_clusters = columnExtraction(re_kernel_kmeans_clusters,0,None)
        the_labels = []
        for img_name in re_kernel_kmeans_clusters:
            the_labels.append(labelExtraction(img_name))

    true_num = 0
    for i in range(len(the_wait_to_labels)):
        evalu = [the_labels[int(idx)] for idx in classification_proba[i]]
        if the_wait_to_labels[i] in evalu:
            true_num = true_num + 1

    accuracy = true_num/len(the_wait_to_labels)

    return accuracy
