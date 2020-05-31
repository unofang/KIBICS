import os
import numpy as np
from tqdm import tqdm

import keras
import keras.applications as kapp
from keras.preprocessing import image
from PIL import Image, ExifTags
import scipy

from dataPreparation.dataClean import fix_orientation,extract_center

def processImages(the_cluster,imgs_dir,preprocess_fn):
    image_size = 224

    image_data = []
    for i in range(len(the_cluster)):
        img = image.load_img(imgs_dir+'/'+the_cluster[i],target_size=(image_size,image_size,3))
        img = fix_orientation(img)
        img = extract_center(img)
        img = img.convert(mode="RGB")
        image_data.append(np.array(img))

    image_data = preprocess_fn(np.array(image_data))
    return image_data

def generateFeatures(model, images):
    return model.predict(images)

def calculateDistances(features):
    # return scipy.spatial.distance.cdist(features, features, "cosine")
    return scipy.spatial.distance.cdist(features, features, "euclidean")

def averageValue(lst):
    return sum(lst) / len(lst)

def similarityForCombination(input_images,imgs_dir,model,preprocess_fn):
    image_data = processImages(input_images,imgs_dir,preprocess_fn)
    features = generateFeatures(model,image_data)
    features = features.reshape(features.shape[0], -1)
    distances = calculateDistances(features)

    sim_img_idx_arr = []
    sim_imgs_arr = []
    sim_total_score_arr = []
    for idx in tqdm(range(len(distances))):
        dist = distances[idx]
        similar_image_indexes = np.argsort(dist)[:4] #3 similar images
        similar_images = [input_images[i] for i in similar_image_indexes]
        total_score = 0
        for x in similar_image_indexes:
            total_score = total_score + dist[x]
        sim_total_score_arr.append(total_score)
        sim_img_idx_arr.append(list(similar_image_indexes))
        sim_imgs_arr.append(similar_images)

    return sim_img_idx_arr,sim_imgs_arr,sim_total_score_arr

def similarityMeasurement(cluster_nums,clusters,the_img,imgs_dir,model,preprocess_fn):
    highest_sim_score = 0
    sim_input_images = [the_img]
    for i in range(len(cluster_nums)):
        sim_input_images.append(clusters[cluster_nums[i]][0])

    for i in range(len(cluster_nums)):
        the_cluster = clusters[cluster_nums[i]][:]
        the_cluster.insert(0,the_img)

        # image_data = processImages(the_cluster,imgs_dir,preprocess_fn)
        #
        # features = generateFeatures(model,image_data)
        # features = features.reshape(features.shape[0], -1)
        # distances = calculateDistances(features)
        # all_dist_to_the_img = distances[0]
        # average_sim_score = averageValue(all_dist_to_the_img)
        # if average_sim_score > highest_sim_score:
        #     highest_sim_score = average_sim_score
        #     the_most_sim_cluster_index = cluster_nums[i]

    image_data = processImages(sim_input_images,imgs_dir,preprocess_fn)
    features = generateFeatures(model,image_data)
    features = features.reshape(features.shape[0], -1)
    distances = calculateDistances(features)
    all_dist_to_the_img = distances[0]
    similar_image_index = np.argsort(all_dist_to_the_img)[1]
    the_most_similar_image = sim_input_images[similar_image_index]
    the_most_sim_cluster_index = int(cluster_nums[similar_image_index-1])

    return the_most_sim_cluster_index

def similarityScore(input_images,imgs_dir,model,preprocess_fn):
    image_data = processImages(input_images,imgs_dir,preprocess_fn)
    features = generateFeatures(model,image_data)
    features = features.reshape(features.shape[0], -1)
    distances = calculateDistances(features)
    dist = distances[0]
    similar_image_indexes = np.argsort(dist)[:4] #3 similar images

    the_index = int(similar_image_indexes[1]) - 1
    return the_index
