from math import exp
from tqdm import tqdm
from datetime import datetime
import time

from dataClustering.dataKMeans import kmeansClustering
from dataPreparation.dataFrame import convertDataFrame

def squaredDistance(vec1, vec2):
    sum = 0
    dim = len(vec1)

    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])

    return sum

def kernel(data, sigma):
    """
    RBF kernel-k-means
    :param data: data points: list of list [[a,b],[c,d]....]
    :param sigma: Gaussian radial basis function
    :return:
    """
    nData = len(data)
    Gram = [[0] * nData for i in range(nData)] # nData x nData matrix
    # TODO
    # Calculate the Gram matrix

    # symmetric matrix
    print ("Start: Transiting each feature into Kernel space")
    for i in tqdm(range(nData)):
        for j in range(i,nData):
            if i != j: # diagonal element of matrix = 0
                # RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
                square_dist = squaredDistance(data[i],data[j])
                base = 2.0 * sigma**2
                Gram[i][j] = exp(-square_dist/base)
                Gram[j][i] = Gram[i][j]
    return Gram

def kernelKMeans(dataset,image_paths,n_clusters,sigma):
    now_start = datetime.now()
    dataset_kernel = kernel(dataset, sigma)
    now_end = datetime.now()
    total_time_gap = int(time.mktime(now_end.timetuple())-time.mktime(now_start.timetuple()))
    print ("Time spent: {} seconds".format(total_time_gap))
    write_kernel_feature_dir = './output/features_extraction_kernel.csv'
    data_save = convertDataFrame(dataset_kernel,image_paths,write_kernel_feature_dir)
    clusters = kmeansClustering(dataset_kernel,image_paths,n_clusters)

    return clusters
