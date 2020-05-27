import os
from tqdm import tqdm

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from pandas import DataFrame as DF

def named_model(model_name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if model_name == 'resnet50':
        return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    elif model_name == 'xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    elif model_name == 'vgg16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    elif model_name == 'vgg19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    elif model_name == 'inceptionv3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    elif model_name == 'mobilenet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    else:
        raise ValueError('Unrecognised model: "{}"'.format(model_name))

def extract_feature(img,model):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    np_features = model.predict(img)[0]

    return np.char.mod('%f', np_features)

def mainDataProcessing(model_name, image_data):
    print("Start: Extracting features")
    model = named_model(model_name)

    features = returnFeatures(image_data,model)
    img_names = returnImageData(image_data)
    # features = []
    # img_names = []
    # for i in tqdm(range(len(image_data))):
    #     # features.append(extract_feature(image_data[i][0],model))
    #     # img_names.append(image_data[i][1])
    #     yield extract_feature(image_data[i][0],model)
    #     yield image_data[i][1]

    return features,img_names

def returnFeatures(image_data,model):
    for i in tqdm(range(len(image_data))):
        yield extract_feature(image_data[i][0],model)

def returnImageData(image_data):
    for i in tqdm(range(len(image_data))):
        yield image_data[i][1]
