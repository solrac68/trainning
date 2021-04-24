from tensorflow import keras
from joblib import dump, load
import config
#from . import config
_PATH_DATA = config._PATH_DATA
_PATH_MODELO = config._PATH_MODELO  

        
def loadModels(path, path2):
    new_model = keras.models.load_model(path)
    bow_model = load(path2)
    return (new_model,bow_model)

def prediccion(opiniones, new_model, bow_model):
    opn = bow_model.transform(opiniones)
    predicciones = new_model.predict(opn)
    return zip(opiniones, predicciones)


if __name__ == "__main__":

    (new_model,bow_model) = loadModels(config._PATH_MODELO_NET , config._PATH_MODELO)

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


    
    
    