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

from dataPreparation.dataClean import fix_orientation,extract_center
from dataPreparation.dataListToFile import saveListToFile

class DataPreparation:
    IMG_EXTS = ['JPG', 'jpg', 'jpeg', 'bmp', 'png']

    def __init__( self, input_dir ):
        self.input_dir = input_dir

    def buildcsv( self ):


        with open('label_train.csv', 'w', newline='') as thecsvfile:
            writer = csv.writer(thecsvfile, quoting=csv.QUOTE_ALL)
            theHeadRow = allPlantDiseasePaths[:]
            # make theHeadRow a new copy of allPlantDiseasePaths
            theHeadRow.insert(0, "filename")
            writer.writerow(theHeadRow)
            # In Python, it is essential to do alignment and indentation making the function or loop a block
            # 在Python语言中，让函数或循环的始末按照规则对齐或缩进形成一个区块是很重要的，否则无法运行

            for index in tqdm(range(len(allPlantDiseasePaths))):
                allThePlantDiseaseImageNames = [f for f in listdir(self.input_dir+"/"+allPlantDiseasePaths[index]) if isfile(join(self.input_dir+"/"+allPlantDiseasePaths[index], f))]
                # print (allThePlantDiseaseImageNames)
                thisRow = [0] * len(allPlantDiseasePaths)
                thisRow[index] = 1
                for theImageName in allThePlantDiseaseImageNames:
                    insertThisRow = thisRow[:]
                    insertThisRow.insert(0,theImageName);
                    writer.writerow(insertThisRow)

        # print ("End: Build the csv")

        return;

    def loadImages( self ):
        print ("Start: Preparing data")
        allPlantDiseasePaths = [];
        for (dirpath, dirnames, filenames) in walk(self.input_dir):
            allPlantDiseasePaths.extend(dirnames)
            break

        class_lookup = []
        train_images_with_filenames = []
        for index in tqdm(range(len(allPlantDiseasePaths))):
            allThePlantDiseaseImageNames = [f for f in listdir(self.input_dir+"/"+allPlantDiseasePaths[index]) if isfile(join(self.input_dir+"/"+allPlantDiseasePaths[index], f))]
            class_lookup.append(allThePlantDiseaseImageNames)
            for i in range(len(allThePlantDiseaseImageNames)):
                img = image.load_img(self.input_dir+allPlantDiseasePaths[index]+'/'+allThePlantDiseaseImageNames[i],target_size=(224,224,3))
                img = fix_orientation(img)
                img = extract_center(img)
                img = img.convert(mode="RGB")
                img = np.array(img)
                train_images_with_filenames.append([img,allThePlantDiseaseImageNames[i]])

        write_lookup_dir = './output/class_lookup.csv'
        if os.path.isfile(write_lookup_dir):
            print ("Note: there is already a lookup csv file")
        else:
            class_lookup_df = saveListToFile(class_lookup,write_lookup_dir)

        return train_images_with_filenames,class_lookup
