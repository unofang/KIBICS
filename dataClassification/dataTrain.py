from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataClassification.saveAndLoadModel import saveModel,loadModel
from dataPreparation.dataListToFile import saveListToFile
from dataPreparation.dataClean import fix_orientation,extract_center
from dataPreparation.enrichDataset import enrichDataset
from dataPlot.trainingClassificationPlot import trainingClassificationPlot

def assignImgsFromClusters(clusters,imgs_dir):
    image_size = 256
    image_data_re = []
    label_matrix = []
    for i in range(len(clusters)):
        label_line = [0 for x in range(len(clusters))]
        label_line[i] = 1
        for j in range(len(clusters[i])):
            # img = image.load_img(imgs_dir+'/'+clusters[i][j],target_size=(image_size,image_size,3))
            img = cv2.imread(imgs_dir+'/'+clusters[i][j])
            img = cv2.resize(img, (image_size, image_size))
            # img = fix_orientation(img)
            # img = extract_center(img)
            # img = image.img_to_array(img)
            # img = img/255
            # image_data_re.append(img)
            # label_matrix.append(label_line)
            image_set = list(enrichDataset(img))
            for k in range(len(image_set)):
                img_x = image.img_to_array(image_set[k])
                img_x = img_x/255
                image_data_re.append(img_x)
                label_matrix.append(label_line)

    image_data_re = np.array(image_data_re)
    # image_data_re.reshape(len(image_data_re),image_size,image_size,3)
    label_matrix = np.array(label_matrix)
    return image_data_re,label_matrix


def trainModel(clusters,imgs_dir):
    image_size = 256
    # existing_model_dir = './output/model/model.json'
    # if os.path.isfile(existing_model_dir):
    #     model = loadModel()
    # else:
    print ("Start: Training the model")
    X,y = assignImgsFromClusters(clusters,imgs_dir)
    print (X.shape)
    print (y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(image_size,image_size,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(image_size,image_size,3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(len(clusters), activation='sigmoid'))
    model.add(Dense(len(clusters), activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), batch_size=32)


    acc_dir = './output/classification_training_acc_plot.png'
    loss_dir = './output/classification_training_loss_plot.png'
    trainingClassificationPlot(history,acc_dir,loss_dir)

        # saveModel(model)

    classes = [str(i) for i in range(len(clusters))]

    return model,classes
