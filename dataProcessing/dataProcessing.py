import os
from tqdm import tqdm

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from pandas import DataFrame as DF

from dataPreparation.dataClean import fix_orientation,extract_center

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

def extract_feature(img_name,model,imgs_dir):
    img = image.load_img(imgs_dir+'/'+img_name,target_size=(224,224,3))
    img = fix_orientation(img)
    img = extract_center(img)
    img = img.convert(mode="RGB")
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    np_features = model.predict(img)[0]

    return np.char.mod('%f', np_features)

def mainDataProcessing(model_name,img_names,imgs_dir):
    print("Start: Extracting features")
    model = named_model(model_name)

    for i in tqdm(range(len(img_names))):
        yield extract_feature(img_names[i],model,imgs_dir)

def mainDataProcessingNotqdm(model_name,img_names,imgs_dir):
    model = named_model(model_name)

    for i in range(len(img_names)):
        yield extract_feature(img_names[i],model,imgs_dir)
