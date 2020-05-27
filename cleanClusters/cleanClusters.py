
def findMaxLength(lst):
    maxLength = 0
    for i in range(len(lst)):
        if len(lst[i])>maxLength:
            maxLength = len(lst[i])
            maxList = lst[i]
            maxListIndex = i

    return maxList, maxLength, maxListIndex

def findAverageLength(lists):
    tol_length = 0
    for list in lists:
        tol_length = tol_length + len(list)

    averageLength = tol_length/len(lists)

    return int(averageLength)

def cleanClusters(clusters):
    maxList, maxLength, maxListIndex = findMaxLength(clusters)
    del clusters[maxListIndex]
    # averageLength = findAverageLength(clusters)
    # newList = maxList[0:averageLength]
    re_cluster_imgs = maxList[:]
    # clusters.insert(maxListIndex,newList)

    return re_cluster_imgs, clusters
