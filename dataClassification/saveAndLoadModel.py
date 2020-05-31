from keras.models import model_from_json
import numpy
import os

def saveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./output/model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./output/model/model.h5")
    print("Saved model to disk")

def loadModel():
    # load json and create model
    json_file = open('./output/model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./output/model/model.h5")
    print("Loaded model from disk")
    return loaded_model
