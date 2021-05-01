from tensorflow import keras
from joblib import dump, load
import config
import json
import dto
import bucket

        
def loadModels(path, path2):
    new_model = keras.models.load_model(path)
    bow_model = load(path2)
    return (new_model,bow_model)

def prediccion(opiniones, new_model, bow_model):
    opn = bow_model.transform(opiniones)
    predicciones = new_model.predict(opn)
    return zip(opiniones, predicciones)

def opinionGen(idmodel,opiniones):
    # Consulta información del modelo desde lata bla modelos
    (bucketName, file_model_mlp, file_model) = getInfoModel(idmodel)
    print("bucket: ", bucket)
    print("file_model_mlp: ", file_model_mlp)
    print("file_model: ", file_model)

    # Descarga modelo mlp desde S3
    download_file_model_mlp = bucket.download_file(bucketName,file_model_mlp)
    # Descarga modelo desde S3
    download_file_model = bucket.download_file(bucketName,file_model)
    print("download_file_model_mlp: ", download_file_model_mlp)
    print("download_file_model: ", download_file_model)

    # Leé los modelos desde las rutas descargadas.
    (new_model,bow_model) = loadModels(download_file_model_mlp , download_file_model)

    
    # Ejecuta el modelo
    predicciones = prediccion(opiniones,new_model,bow_model)
    # Ordena los resultados
    resultados = [("Positivo" if pr > 0.5 else "Negativo",op) for op, pr in predicciones]
    jsonStr = json.dumps(resultados)

    print("jsonStr: ", jsonStr)

    #[print(f"'{x}' es '{y}'") for (x,y) in resultados]
    #opi = [("Lenovo Thinkpad es una excelente máquina, la recomiendo","Positivo :)"),("Las camas estaban mal arregladas","Negativo :(")]
    #jsonStr = json.dumps(opi)
    #print(jsonStr)

    estado = False

    # return {
    #     "statusCode": 200,
    #     "headers":{
    #         'myHeader':'test'
    #     },
    #     "body": event["body"],
    #     "isBase64Encoded": estado
    # }
    return {
        "statusCode": 200,
        "headers":{
            'myHeader':'test'
        },
        "body": jsonStr,
        "isBase64Encoded": estado
    }

def getDatos(event, context):
    # TODO implement
    
    ## ESQUEMA event["body"]##
    # {
    #   "$schema": "http://json-schema.org/draft-04/schema#",
    #   "title": "Textos",
    #   "type": "object",
    #   "properties": {
    #     "idmodel": { "type": "string" },
    #     "opiniones": {
    #       "type": "array",
    #       "items": { "type": "string" }
    #     }
    #   }
    # }
    
    payload = json.loads(event["body"])
    # {'idmodel': 'Ingesta11.txt', 'opiniones': ['Lenovo Thinkpad es una excelente máquina, la recomiendo.', 'Las camas estaban mal arregladas']}

    #print(type(payload))
    print("payload: ", payload)
    #print("idmodel:", payload['idmodel'])

    idmodel = payload['idmodel']
    print("idmodel:", idmodel)
    # Listado de opiniones
    opiniones = payload['opiniones']
    print("opiniones:", opiniones)

    return (idmodel, opiniones)


def handler(event, context):
    (idmodel, opiniones) = getDatos(event, context)
    return opinionGen(idmodel,opiniones)

def getInfoModel(idModel:str):
    bucket = None
    file_model_mlp = None
    file_model = None
    nombreTabla = config._TABLE_MODELS
    key = "idFile"

    respuesta = dto.query(nombreTabla,key,idModel)
    
    for obj in respuesta:
        bucket = obj['info']['bucket']
        file_model_mlp = obj['info']['file_model_mlp']
        file_model = obj['info']['file_model']

    return (bucket,file_model_mlp,file_model)

if __name__ == "__main__":

    opiniones = ["Lenovo Thinkpad es una excelente máquina, la recomiendo.", 
    "Iphone no es tan bueno como dicen, por lo menos eso creo para el precio tan alto que tiene.",
    "Las camas estaban mal arregladas",
    "El ventilador no funcionaba",
    "La comida fue excelente"]
    idmodel = "DB_proyecto_SA.txt"
    resultados = opinionGen(idmodel,opiniones)

    print(resultados)

    
    
    