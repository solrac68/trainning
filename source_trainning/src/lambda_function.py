import json
import boto3
import uuid
import csv
from urllib.parse import unquote_plus

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('data2')

# def readFile(path, file):
#     with open(path) as f:
#         reader = csv.reader(f, delimiter='\t')
#         data = [(str(uuid.uuid4()),file,col1, int(col2))
#                 for col1, col2 in reader]
#     return data
    
def readFile(path, file):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [(file,str(uuid.uuid4()),col1, col2)
                for col1, col2 in reader]
    return data
    
# def copyToDynamoDB(data):
#     print("Tamaño archivo: ", len(data))
#     with table.batch_writer() as batch:
#         for i in data:
#             for j in i:
#                 batch.put_item(
#                     Item={
#                         'idFile': j[0],
#                         'uuid4': j[1],
#                         'texto': j[2],
#                         'calificacion': j[3]
#                     }
#                 )

def copyToDynamoDB(data):
        for i in data:
            print("Tamaño archivo: ", len(i))
            for j in i:
                table.put_item(
                    Item={
                        'idFile': j[0],
                        'uuid4': j[1],
                        'texto': j[2],
                        'calificacion': j[3]
                    }
                )


def downloadFromBucket(bucket, key):
    ubicaciones = []
    tmpkey = key.replace('/', '')
    download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
    print(download_path)
    s3_client.download_file(bucket, tmpkey, download_path)
    ubicaciones.append((download_path,key))
    return ubicaciones

def lambda_handler(bucket, key):
    listaUbicacionesLocales = downloadFromBucket(bucket, key)
    print(listaUbicacionesLocales)
    data = [readFile(path,file) for path,file in listaUbicacionesLocales]
    copyToDynamoDB(data)
    print("Copiado en Dynamo")

if __name__ == "__main__":
    bucket = "files-files-training"
    key = "Ingesta11.txt"
    lambda_handler(bucket, key)
