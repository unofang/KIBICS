import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import psutil
import os
import gc

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
from dataPreparation.dataClean import fix_orientation,extract_center
from keras.regularizers import l2
from keras.optimizers import Adam

from dataPreparation.dataListToFile import saveListToFile
from dataPlot.trainingClassificationPlot import trainingClassificationPlot

from memoryManagement.memoryRelease import releaseList
from memoryManagement.memoryCheck import memoryCheck

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

def create_white_image():
    img = np.zeros([112,112,3],dtype=np.uint8)
    img.fill(255)
    return img

def read_image(filename):

    size = 2
    #first we read the image, as a raw file to the buffer
    img = image.load_img(filename,target_size=(224,224,3))
    img = fix_orientation(img)
    img = extract_center(img)
    img = image.img_to_array(img)
    img = img/255
    img = img[::size, ::size]

    #then we convert the image to numpy array using np.frombuffer which interprets buffer as one dimensional array
    return img

def get_data(total_sample_size, clusters,imgs_dir):
    #read the image
    image = read_image(imgs_dir + '/' + clusters[0][0])
    #get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    dim3 = image.shape[2]

    count = 0

    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, dim1, dim2, dim3]) # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])

    the_clusters_num = len(clusters)

    the_minimum_size = len(clusters[0])

    for i in range(the_clusters_num):
        if the_minimum_size > len(clusters[i]):
            the_minimum_size = len(clusters[i])

    for i in tqdm(range(the_clusters_num)):
        for j in range(int(total_sample_size/the_clusters_num)):
            ind1 = 0
            ind2 = 0

            while ind1 == ind2:
                ind1 = np.random.randint(the_minimum_size)
                ind2 = np.random.randint(the_minimum_size)

            img1 = read_image(imgs_dir + '/' + clusters[i][ind1])
            img2 = read_image(imgs_dir + '/' + clusters[i][ind2])

            #store the images to the initialized numpy array
            x_geuine_pair[count, 0, :, :, :] = img1
            x_geuine_pair[count, 1, :, :, :] = img2

            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, dim1, dim2, dim3])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in tqdm(range(int(total_sample_size/the_minimum_size))):
        for j in range(the_minimum_size):

            #read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(the_clusters_num)
                ind2 = np.random.randint(the_clusters_num)
                if ind1 != ind2:
                    break

            img1 = read_image(imgs_dir + '/' + clusters[ind1][j])
            img2 = read_image(imgs_dir + '/' + clusters[ind2][j])

            x_imposite_pair[count, 0, :, :, :] = img1
            x_imposite_pair[count, 1, :, :, :] = img2
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1

    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y

def getSiameseModel(left_input,right_input):
    """
        Model architecture
    """
    input_shape = (112,112,3)
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    initialize_weights = 'random_normal'
    initialize_bias = 'zeros'

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (9,9), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (5,5), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3,3), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (2,2), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam(0.001, decay=2.5e-4)
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    # return the model
    return siamese_net

def siameseNetworkTrain(clusters,imgs_dir):
    # tfback._get_available_gpus = _get_available_gpus
    # tfback._get_available_gpus()
    # tf.config.list_logical_devices()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    total_sample_size = 20000

    X, Y = get_data(total_sample_size, clusters,imgs_dir)

    memoryCheck()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

    releaseList(X)
    releaseList(Y)
    gc.collect()

    memoryCheck()

    input_dim = x_train.shape[2:]
    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

    model = getSiameseModel(img_a, img_b)

    img_1 = x_train[:, 0]
    img_2 = x_train[:, 1]

    memoryCheck()

    history = model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2, epochs=15)

    memoryCheck()

    releaseList(img_1)
    releaseList(img_2)
    releaseList(y_train)
    gc.collect()

    memoryCheck()

    acc_dir = './output/siamese_training_acc_plot.png'
    loss_dir = './output/siamese_training_loss_plot.png'
    trainingClassificationPlot(history,acc_dir,loss_dir)

    return model

def theSort(sub_li):
    sub_li.sort(key = lambda x: x[0])
    return sub_li

def getAllImages(clusters,imgs_dir):
    for i in range(len(clusters)):
        yield getTheRowImages(clusters[i],imgs_dir)

def getTheRowImages(this_cluster,imgs_dir):
    images = []
    for i in range(len(this_cluster)):
        images.append(read_image(imgs_dir+'/'+this_cluster[i]))
    return images

def siameseNetworkPredict(model,imgs_dir,clusters_a,clusters_b):
    all_images_a = list(getAllImages(clusters_a,imgs_dir))
    all_images_b = list(getAllImages(clusters_b,imgs_dir))

    allProbs = []
    for i in tqdm(range(len(all_images_a))):
        the_highest_score = 0
        inputs_a = all_images_a[i]
        for j in range(len(all_images_b)):
            inputs_b = all_images_b[j]
            if len(inputs_a)>len(inputs_b):
                for l in range(len(inputs_a)-len(inputs_b)):
                    # inputs_b.append(inputs_b[0])
                    inputs_b.append(create_white_image())
            else:
                for l in range(len(inputs_b)-len(inputs_a)):
                    # inputs_a.append(inputs_a[0])
                    inputs_a.append(create_white_image())

            probs = model.predict([inputs_a,inputs_b])
            probs = [x[0] for x in probs]
            the_score = sum(probs)/len(probs)
            if the_score > the_highest_score:
                the_highest_score = the_score
                the_index_a = i
                the_index_b = j

        allProbs.append([the_highest_score,the_index_a,the_index_b])

    return allProbs

def siameseNetworkMain(imgs_dir,clusters_a,clusters_b):
    memoryCheck()

    model = siameseNetworkTrain(clusters_a,imgs_dir)

    gc.collect()
    memoryCheck()

    predict = siameseNetworkPredict(model,imgs_dir,clusters_a,clusters_b)

    gc.collect()
    memoryCheck()

    write_test_predict_dir = './output/test_predict.csv'
    saveListToFile(predict,write_test_predict_dir)

    return predict
