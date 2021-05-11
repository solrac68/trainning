import json
import boto3
import uuid
import csv
from urllib.parse import unquote_plus
import config

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(config._TABLE_INGEST)


    
def readFile(path, file):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [(file,str(uuid.uuid4()),col1, col2)
                for col1, col2 in reader]
    return data
    
def copyToDynamoDB(data):
    print("Tamaño archivo: ", len(data))
    with table.batch_writer() as batch:
        for i in data:
            for j in i:
                batch.put_item(
                    Item={
                        'idFile': j[0],
                        'uuid4': j[1],
                        'texto': j[2],
                        'calificacion': j[3]
                    }
                )


def downloadFromBucket(event):
    ubicaciones = []
    print(event)
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        tmpkey = key.replace('/', '')
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
        print("bucket: ",bucket)
        print("key: ",key)
        print("download_path: ",download_path)
        s3_client.download_file(bucket, key, download_path)
        ubicaciones.append((download_path,key))
    return ubicaciones

def lambda_handler(event, context):
    listaUbicacionesLocales = downloadFromBucket(event)
    print(listaUbicacionesLocales)
    data = [readFile(path,file) for path,file in listaUbicacionesLocales]
    copyToDynamoDB(data)
    return {
        'statusCode': 200,
        'body': {
            'tablaDynamodb': config._TABLE_INGEST
        }
    }

if __name__ == "__main__":
    bucket = "files-files-training"
    key = "Ingesta11.txt"
    lambda_handler(bucket, key)
