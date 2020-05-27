import csv

def readcsvToList(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data
