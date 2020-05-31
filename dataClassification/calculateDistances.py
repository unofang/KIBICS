import scipy

def calculateDistances(features):
    return scipy.spatial.distance.cdist(features, features, "cosine")
