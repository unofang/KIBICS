import numpy as np

def dataCorrection(features_df):
    dataset = features_df.values
    dataset = [[float(y) for y in x] for x in dataset]
    dataset = np.array(dataset)

    return dataset
