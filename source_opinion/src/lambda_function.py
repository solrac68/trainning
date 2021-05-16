import config
import json
import dto

def lambda_handler(event, context):
    # TODO implement
    nombreTabla = config._TABLE_MODELS
    columna = "idFile"
    resultados = dto.queryKeys(nombreTabla, columna)

    jsonStr = json.dumps(resultados)

    return {
        "statusCode": 200,
        "headers":{
            'myHeader':'test'
        },
        "body": jsonStr,
        "isBase64Encoded": True
    }