import os
import numpy as np
from pandas import DataFrame as DF

from dataPreparation.dataLoad import dataLoadFromFile

def buildLookupDict(class_lookup_df):
    # print (class_lookup_df)
    label_lookup = {}
    for i in range(len(class_lookup_df)):
        for j in range(len(class_lookup_df[i])):
            file_name = class_lookup_df[i][j]
            label_lookup[file_name] = i

    return label_lookup
