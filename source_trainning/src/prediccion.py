from tensorflow import keras
from joblib import dump, load
import config
#from . import config
#_PATH_DATA = config._PATH_DATA
#_PATH_MODELO = config._PATH_MODELO  

        
def loadModels(path, path2):
    new_model = keras.models.load_model(path)
    bow_model = load(path2)
    return (new_model,bow_model)

def prediccion(opiniones, new_model, bow_model):
    opn = bow_model.transform(opiniones)
    predicciones = new_model.predict(opn)
    return zip(opiniones, predicciones)

import json

def handler(event, context):
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

    print(type(payload))
    print(payload)
    print("idmodel:", payload['idmodel'])
    print("opiniones:", payload['opiniones'])
    
    [print(obj) for obj in payload['opiniones']]

    estado = False
    
    return {
        "statusCode": 200,
        "headers":{
            'myHeader':'test'
        },
        "body": event["body"],
        "isBase64Encoded": estado
    }



if __name__ == "__main__":

    path_modelo = "/tmp/daaad05a-ab10-49e3-8500-4acc651713e0_model.joblib"
    path_modelo_net = "/tmp/1f9cedd1-9dc3-4556-89ae-38adebc6d3ed_modelmlp.h5"
    (new_model,bow_model) = loadModels(path_modelo_net , path_modelo)

    

    opiniones = ["Lenovo Thinkpad es una excelente máquina, la recomiendo.", 
    "Iphone no es tan bueno como dicen, por lo menos eso creo para el precio tan alto que tiene.",
    "Las camas estaban mal arregladas",
    "El ventilador no funcionaba",
    "La comida fue excelente"]

    predicciones = prediccion(opiniones,new_model,bow_model)


    print("#########################################################################")
    print("############################# Predicción ################################")
    print("#########################################################################")

    resultados = [(op, "Positivo :)" if pr > 0.5 else "Negativo :(") for op, pr in predicciones]

    [print(f"'{x}' es '{y}'") for (x,y) in resultados]


    
    
    