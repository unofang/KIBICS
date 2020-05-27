import csv

from pandas import DataFrame as DF

def convertDataFrame(data,first_col,write_dir):
    data = DF(data, dtype=object)
    if first_col != None:
        first_col = DF(first_col, dtype=str)
        data.insert(0, 'ID', first_col)

    data.to_csv(write_dir, index=False)

    return data
