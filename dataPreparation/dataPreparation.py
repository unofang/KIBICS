import os
from os import listdir
from os.path import isfile, join
from os import walk
import csv
from tqdm import tqdm

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from pandas import DataFrame as DF

from dataPreparation.dataListToFile import saveListToFile

class DataPreparation:
    IMG_EXTS = ['JPG', 'jpg', 'jpeg', 'bmp', 'png']

    def __init__( self, input_dir ):
        self.input_dir = input_dir

    def loadImages( self ):
        print ("Start: Preparing data")
        allPlantDiseasePaths = [];
        for (dirpath, dirnames, filenames) in walk(self.input_dir):
            allPlantDiseasePaths.extend(dirnames)
            break

        class_lookup = []
        for index in tqdm(range(len(allPlantDiseasePaths))):
            allThePlantDiseaseImageNames = [f for f in listdir(self.input_dir+"/"+allPlantDiseasePaths[index]) if isfile(join(self.input_dir+"/"+allPlantDiseasePaths[index], f))]
            class_lookup.append(allThePlantDiseaseImageNames)

        write_lookup_dir = './output/class_lookup.csv'
        if os.path.isfile(write_lookup_dir):
            print ("Note: there is already a lookup csv file")
        else:
            class_lookup_df = saveListToFile(class_lookup,write_lookup_dir)

        for index in range(len(allPlantDiseasePaths)):
            allThePlantDiseaseImageNames = [f for f in listdir(self.input_dir+"/"+allPlantDiseasePaths[index]) if isfile(join(self.input_dir+"/"+allPlantDiseasePaths[index], f))]
            for i in range(len(allThePlantDiseaseImageNames)):
                yield allThePlantDiseaseImageNames[i]
