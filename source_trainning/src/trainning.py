from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
import config
import uuid
# Import `Sequential` from `keras.models`
#from keras.models import Sequential
from tensorflow.keras import Sequential

# Import `Dense` from `keras.layers`
#from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Dense, Dropout, Activation

#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time

from tensorflow import keras
import json

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


#_PATH_DATA = config._PATH_DATA
#_PATH_MODELO = config._PATH_MODELO

def error_measures(Yestimado, Yteorico):
    
    CM = confusion_matrix(Yteorico, Yestimado)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    precision = TP / (TP + FP)
    recall  = TP / (TP + FN)
    f1score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1score
    
def classification_error(y_est, y_real):
    err = 0
    for y_e, y_r in zip(y_est, y_real):
      #Notar que y_est y y_real son matrices One-Hot Encoded (y_est tambien requiere un round para llevarla a binaria)
      #Por lo tanto se debe hacer la decodificación de estas matrices para comparar los valores predichos y los
      #Teóricos. Note que en una predicción correcta deben coindidir las posiciones de los arreglos que estén
      #activadas (bit 1 o hot :)!)
      #pos_y_e = np.argmax(y_e)
      #pos_y_r = np.argmax(y_r)
      #print "Pos predicho: " + str(pos_y_e)
      #print "Pos teórico: " + str(pos_y_r)
      if (y_e) != y_r:
          err += 1


    return err/np.size(y_real,0)
  
  
def carga_datos(path):
    df = pd.read_csv(path, delimiter = "\t", error_bad_lines=False,header=None)
    df.columns=['texto','sentimiento']
    X=df.iloc[:,0]
    y=df.iloc[:,1]
    y=np.asarray(y)
    
    return (X,y)
    
def modeloBagOfWords(X, path):
    vector=CountVectorizer(ngram_range=(1, 2))
    modelo = vector.fit(X)
    bagOfWords=modelo.transform(X)
    bagOfWords=bagOfWords.toarray()
    dump(modelo, path)
    
    return bagOfWords

def trainningNetwork(bagOfWords,y):
    tiempo_i = time.time()

    Errores = np.ones(10)
    # Sens = np.zeros(10)
    # Espec = np.zeros(10)
    Precision= np.zeros(10)
    Recall= np.zeros(10)
    F1score = np.zeros(10)
    j = 0
    kf = KFold(n_splits=10, shuffle=True)
    
    
    for train_index, test_index in kf.split(bagOfWords):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = bagOfWords[train_index], bagOfWords[test_index]
        y_train, y_test = y[train_index], y[test_index] 
    
        #Instanciamos el modelo MLP
        model = Sequential()
    
        model.add(Dense(units=15, activation='relu', input_dim=bagOfWords.shape[1]))
    
        #Dropout
        model.add(Dropout(0.25))
        
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=20, activation='relu'))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=40, activation='relu'))
        model.add(Dense(units=40, activation='relu'))
        model.add(Dense(units=20, activation='relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))  
    
      
        # Model config
        model.get_config()
    
        # List all weight tensors 
        model.get_weights()
    
        model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
        #Train process
        model.fit(X_train, y_train, epochs=30)
        # dump(model, 'mlpModel.joblib')
        # Test
        ypred = model.predict(X_test)
        y_pred = []
        for yp,yt in zip(ypred,y_test):
          
          if yp <= 0.5:
            yp=0
          else:
            yp=1
          y_pred.append(yp)
          #print(yp, '\t', yt)
        
        y_pred = np.asarray(y_pred)
        
        Errores[j] = classification_error(y_pred, y_test)
        #print('Error en la iteración: ', Errores[j])
        
        precision, recall, f1score = error_measures(y_pred,y_test)
        # Sens[j] = sens
        # Espec[j] = esp
        Precision[j] = precision
        Recall[j] = recall
        F1score[j] = f1score
        j+=1
        
    return model,Errores,Precision,Recall,F1score,time.time()-tiempo_i
    
def saveModel(path,model):
    model.save(path)
        
def loadModel(path):
    model = keras.models.load_model(path)
    return model

# def lambda_handler((event,context):
#     print(f"Evento: {event}")

def handler(event, context):
    return main_handler1(event, context)

def main_handler2(event, context):
    ### No borrar #######
    for record in event['Records']:
        print("test")
        #payload = record['body']
        payload = json.loads(record["body"])
        print(str(payload))
        for r in payload['requestPayload']['Records']:
            name = r['s3']['bucket']['name']
            key = r['s3']['object']['key']
            print(f"El nombre del bucket es : {name}")
            print(f"El nombre de la clave es : {key}")
        
        nombreTabla = payload['responsePayload']['body']['tablaDynamodb']
        print(f"El nombre de la tabla es: {nombreTabla}")
        
    return {
        'statusCode': 200,
        'body': {
            'name':name,
            'key':key,
            'tabla':nombreTabla
        }  
    }


def main_handler1(event, context):
    keys = []
    print(event)
    logger.info(f'Evento:{event}')

    (Errores,Precision,Recall,F1score,lapso_tiempo,path_model,path_model_net) = main()
        
    return {
        'statusCode': 200,
        'body': {
            'path_model':path_model,
            'path_model_net':path_model_net,
            'lapso_tiempo':str(lapso_tiempo),
            'mediaPrecision':str(np.mean(Precision))
        }  
    }



def main():
    tiempo_i = time.time() 
    (X,y) = carga_datos(config._PATH_DATA)
    path_model = '/tmp/{}{}'.format(uuid.uuid4(), config._MODELO)
    bagOfWords = modeloBagOfWords(X,path_model)
    (model,Errores,Precision,Recall,F1score,tiempoEjec) = trainningNetwork(bagOfWords,y)
    path_model_net = '/tmp/{}{}'.format(uuid.uuid4(), config._MODELO_NET)
    saveModel(path_model_net,model)
    lapso_tiempo = time.time() - tiempo_i
    return (Errores,Precision,Recall,F1score,lapso_tiempo,path_model,path_model_net)

if __name__ == "__main__":
    tiempo_i = time.time()   
    (Errores,Precision,Recall,F1score,tiempoEjec,path_model,path_model_net) = main()   

    print("\nPrecision: " + str(np.mean(Precision)) + " +/- " + str(np.std(Precision)))
    lapso_tiempo = time.time() - tiempo_i
    print(f"Modelo almacenado en: {path_model_net} tiempo total: {lapso_tiempo} s")




    
    
    