import boto3
import uuid
import csv


# Let's use Amazon S3
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('data2')

def readFile(path, file):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [(file,str(uuid.uuid4()),col1, int(col2))
                for col1, col2 in reader]
    return data

def downloadFromBucket():
    ubicaciones = []
    for bucket in s3.buckets.all():
        print(bucket.name)
        for obj in bucket.objects.all():
            tmpkey = obj.key.replace('/', '')
            download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
            print("bucket.name: ", bucket.name)
            print("obj.key: ", obj.key)
            s3_client.download_file(bucket.name, obj.key, download_path)
            ubicaciones.append((download_path,obj.key))
            print(download_path)
    return ubicaciones

def downloadFromBucket2(bucketname, objkey):
    ubicaciones = []
    tmpkey = objkey.replace('/', '')
    download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
    #print("bucket.name: ", bucket.name)
    #print("obj.key: ", obj.key)
    s3_client.download_file(bucketname, objkey, download_path)
    ubicaciones.append((download_path,objkey))
    print(download_path)
    return ubicaciones

def copyToDynamoDB(data):
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
    
    


if __name__ == "__main__":
    #listaUbicacionesLocales = downloadFromBucket()
    bucketname='files-training'
    objkey='Ingesta11.txt'
    listaUbicacionesLocales = downloadFromBucket2(bucketname, objkey)
    print(listaUbicacionesLocales)
    data = [readFile(path,file) for path,file in listaUbicacionesLocales]
    copyToDynamoDB(data)
    print("Copiado en dynamoDb")

    