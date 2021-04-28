import csv
import uuid
import config


def readFile(path, file):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [(file,str(uuid.uuid4()),col1, col2)
                for col1, col2 in reader]
    return data

if __name__ == "__main__":
    #path = config._PATH_DATA
    path = "Datasets/Ingesta9.txt"
    file = "Ingesta9.txt"
    data = readFile(path,file)
    print(len(data))
    for d in data:
        print(d[2],d[3])
    #print(data[0])


