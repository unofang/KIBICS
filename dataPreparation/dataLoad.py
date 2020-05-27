import os
import pandas as pd
import csv

def dataLoadFromFile(the_feature_extract_file):
    assert os.path.exists(the_feature_extract_file),\
        'Parsing data: File not found: "{}"'.format(the_feature_extract_file)
    assert the_feature_extract_file.lower().endswith('.csv'),\
        'Parsing data: Requires a .csv file'
    features_df = pd.read_csv(the_feature_extract_file, dtype=object)

    return features_df

def dataFileToList(the_feature_extract_file):
    with open(the_feature_extract_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data
