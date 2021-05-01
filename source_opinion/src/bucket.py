import boto3
import botocore
from botocore.exceptions import ClientError
import config
import logging
#dynamodb = boto3.resource('dynamodb')
#table = dynamodb.Table('data2')
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

def download_file(bucket,file):
    download_path = '/tmp/{}'.format(file)
    s3_client.download_file(bucket, file, download_path)

    return download_path

def upload_file(file_name, bucket, object_name=None):

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload(files:[]):
    #file_model = "ingesta11.txt"
    for file in files:
        path_model = f"Datasets/{file}"
        upload_file(path_model, config._BUCKET, object_name=file)
        print(f"Archivo {file} subido a {config._BUCKET}")

def deleteObjects():
    for bucket in s3.buckets.all():
        print("bucket.name: ", bucket.name)
        for obj in bucket.objects.all():
            print(f"---> Borrando {obj.key} ")
            obj.delete()

def consultando():
    for bucket in s3.buckets.all():
        print("bucket.name: ", bucket.name)
        for obj in bucket.objects.all():
            print(f"---> Objeto {obj.key} ")

def test1():
    #deleteObjects()
    #upload(["DB_proyecto_SA.txt"])
    #upload(["Ingesta11.txt","DB_proyecto_SA.txt"])
    upload(["Ingesta12.txt"])
    

if __name__ == '__main__':
    #test1()
    consultando()

    
    
