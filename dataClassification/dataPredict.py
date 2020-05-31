from keras.preprocessing import image
import keras.applications as kapp

import numpy as np
from tqdm import tqdm

from dataClassification.similarityMeasurement import similarityMeasurement,similarityScore
from dataPreparation.dataListToFile import saveListToFile
from dataPreparation.dataClean import fix_orientation,extract_center
from dataPreparation.dataListToFile import saveListToFile

def buildModel(model_name):
    """ Create a pretrained model without the final classification layer. """
    if model_name == "resnet50":
        model = kapp.resnet50.ResNet50(weights="imagenet", include_top=False)
        return model, kapp.resnet50.preprocess_input
    elif model_name == "vgg16":
        model = kapp.vgg16.VGG16(weights="imagenet", include_top=False)
        return model, kapp.vgg16.preprocess_input
    elif model_name == 'xception':
        model = kapp.xception.Xception(weights="imagenet", include_top=False)
        return model, kapp.xception.preprocess_input
    elif model_name == 'vgg19':
        model = kapp.vgg19.VGG19(weights="imagenet", include_top=False)
        return model, kapp.vgg19.preprocess_input
    elif model_name == 'inceptionv3':
        model = kapp.inception_v3.InceptionV3(weights="imagenet", include_top=False)
        return model, kapp.inception_v3.preprocess_input
    elif model_name == 'mobilenet':
        model = kapp.mobilenet.MobileNet(weights="imagenet", include_top=False)
        return model, kapp.mobilenet.preprocess_input
    else:
        raise Exception("Unsupported model error")

def predictTheImg(model,classes,imgs_dir,the_imgs,clusters,sim_model_name):
    image_size = 256
    to_predict_imgs = []
    print ("Start: Loading to-be-predicted images")
    for i in tqdm(range(len(the_imgs))):
        img = image.load_img(imgs_dir+'/'+the_imgs[i],target_size=(image_size,image_size,3))
        img = fix_orientation(img)
        img = extract_center(img)
        img = image.img_to_array(img)
        img = img/255
        to_predict_imgs.append(img.reshape(image_size,image_size,3))

    to_predict_imgs = np.array(to_predict_imgs)
    print ("Start: Predicting images into clusters")
    proba = model.predict(to_predict_imgs)
    proba_dir = './output/classification_proba.csv'
    saveListToFile(np.argsort(proba),proba_dir)
    # print (np.argsort(proba))

    print ("Start: Adding images into predicted clusters")

    sim_model, preprocess_fn = buildModel(sim_model_name)

    top_x_array = []
    for i in tqdm(range(len(proba))):
        the_x = int(len(clusters)/3)
        if the_x < 3 and len(clusters)>3:
            the_x = 3
        top_x = np.argsort(proba[i])[:-the_x:-1]
        # cluster_index = int(classes[top_x[1]])
        top_x_array.append(top_x)
        cluster_index = similarityMeasurement(top_x,clusters,the_imgs[i],imgs_dir,sim_model,preprocess_fn)
        clusters[cluster_index].append(the_imgs[i])

    write_top_x_array_dir = './output/top_x_array.csv'
    saveListToFile(top_x_array,write_top_x_array_dir)

    for i in range(len(clusters)):
        print("Cluster {} containing {} objects".format(i,len(clusters[i])))

    return clusters

def predictTheImgWithSimilarity(imgs_dir,the_imgs,clusters,sim_model_name):
    sim_model, preprocess_fn = buildModel(sim_model_name)
    input_first_image_name_of_each_cluster = []
    for i in tqdm(range(len(clusters))):
        input_first_image_name_of_each_cluster.append(clusters[i][0])

    for i in tqdm(range(len(the_imgs))):
        input_images = input_first_image_name_of_each_cluster[:]
        input_images.insert(0,the_imgs[i])
        the_cluster_num = similarityScore(input_images,imgs_dir,sim_model,preprocess_fn)
        clusters[the_cluster_num].append(the_imgs[i])

    return clusters

def predictTheImgCombineSiamese(imgs_dir,the_imgs,clusters,sim_model_name):
    image_size = 256
    to_predict_imgs = []
    print ("Start: Loading to-be-predicted images")
    for i in tqdm(range(len(the_imgs))):
        img = image.load_img(imgs_dir+'/'+the_imgs[i],target_size=(image_size,image_size,3))
        img = fix_orientation(img)
        img = extract_center(img)
        img = image.img_to_array(img)
        img = img/255
        to_predict_imgs.append(img.reshape(image_size,image_size,3))

    to_predict_imgs = np.array(to_predict_imgs)
    print ("Start: Predicting images into clusters")
    proba = model.predict(to_predict_imgs)
    proba_dir = './output/classification_proba.csv'
    saveListToFile(np.argsort(proba),proba_dir)
    # print (np.argsort(proba))

    print ("Start: Adding images into predicted clusters")

    sim_model, preprocess_fn = buildModel(sim_model_name)
