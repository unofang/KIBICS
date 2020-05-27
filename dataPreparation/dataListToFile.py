import csv

def saveListToFile(data,write_dir):
    with open(write_dir, 'w', newline='') as thefile:
        writer = csv.writer(thefile, quoting=csv.QUOTE_ALL)
        for i in range(len(data)):
            therow = data[i][:]
            writer.writerow(therow)
