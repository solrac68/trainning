import logging
import boto3
import config
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')

def query(nombreTabla,key,value):
    table = dynamodb.Table(nombreTabla)
    response = table.query(
        KeyConditionExpression=Key(key).eq(value)
    )
    
    return response['Items']

def queryKeys(nombreTabla, columna):
    table = dynamodb.Table(nombreTabla)
    scan_kwargs = {
        'ProjectionExpression': columna
    }
    response = table.scan(**scan_kwargs)
    
    return response['Items']

if __name__ == "__main__":
    nombreTabla = config._TABLE_MODELS
    key = "idFile"
    value = "Ingesta11.txt"
    respuesta = query(nombreTabla,key,value)
    #print(type(respuesta))

    print(respuesta)
    [print(f"Bucket: {obj['info']['bucket']}\nModelo mlp:{obj['info']['file_model_mlp']}") for obj in respuesta]
    
    columna = key
    respuesta = queryKeys(nombreTabla, columna)
    #print(type(respuesta))
    

    print(respuesta)
    [print(f"{obj['idFile']}") for obj in respuesta]