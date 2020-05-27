from sklearn.cluster import SpectralClustering
import numpy as np
from datetime import datetime
import time

def spectralClustering(dataset,image_paths,n_clusters):
    print ("Start: Spectral clustering")
    # now_start = datetime.now()
    predictions = SpectralClustering(n_clusters,
                                     assign_labels="discretize",
                                     eigen_solver='arpack').fit(dataset)
    # assign_labels="discretize",
    # now_end = datetime.now()
    # total_time_gap = int(time.mktime(now_end.timetuple())-time.mktime(now_start.timetuple()))
    clusters = [[] for x in range(n_clusters) ]

    result = predictions.labels_

    for i in range(len(result)):
        clusters[result[i]].append(image_paths[i])

    for i in range(len(clusters)):
        print("Cluster {} containing {} objects".format(i,len(clusters[i])))
    # print ("Time spent: {} seconds".format(total_time_gap))

    return clusters
