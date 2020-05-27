from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime
import time

def kmeansClustering(dataset,image_paths,n_clusters):
    print ("Start: K-means clustering")
    now_start = datetime.now()
    model = KMeans(n_clusters, n_jobs=-1, random_state=728)
    model.fit(dataset)
    now_end = datetime.now()
    total_time_gap = int(time.mktime(now_end.timetuple())-time.mktime(now_start.timetuple()))
    predictions = model.predict(dataset)
    #print(predictions)
    # w, h = 8, 5;
    # Matrix = [[0 for x in range(w)] for y in range(h)]
    clusters = [[] for x in range(n_clusters) ]
    for i in range(len(predictions)):
        clusters[predictions[i]].append(image_paths[i])

    for i in range(len(clusters)):
        print("Cluster {} containing {} objects".format(i,len(clusters[i])))
    print ("Time spent: {} seconds".format(total_time_gap))

    return clusters
