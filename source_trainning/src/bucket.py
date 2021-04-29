import boto3
import botocore
from botocore.exceptions import ClientError
import config
#dynamodb = boto3.resource('dynamodb')
#table = dynamodb.Table('data2')
s3 = boto3.resource('s3')

def deleteObjects():
    for bucket in s3.buckets.all():
        print("bucket.name: ", bucket.name)
        for obj in bucket.objects.all():
            print(f"Borrando {obj.key} ")
            obj.delete()

if __name__ == '__main__':
    deleteObjects()

    
    
