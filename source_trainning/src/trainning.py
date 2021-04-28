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
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

from pandas import DataFrame

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')


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
    print("df")
    print(df.shape)
    X=df.iloc[:,0]
    y=df.iloc[:,1]
    y=np.asarray(y)
    
    return (X,y)

def query_trainning(nombreTabla, file):
    table = dynamodb.Table(nombreTabla)
    response = table.query(
        KeyConditionExpression=Key('idFile').eq(file)
    )
    calificaciones = response['Items']


    print(f"Archivo: {file} en Dynamo: {len(calificaciones)} rows")

    for califi in calificaciones:
        califi['calificacion'] = int(califi['calificacion'])
    return calificaciones

    #df = DataFrame (response['Items'],columns=['texto','calificacion'])
    

def load_data_from_dynamo(nombreTabla, files:[]):
    resultados = [query_trainning(nombreTabla,file) for file in files]
    unionList = []
    for r in resultados:
        unionList += r
    df = DataFrame (unionList,columns=['texto','calificacion'])
    print("df2")
    print(df.shape)
    return df

def carga_datos2(nombreTabla, files:[]):
    df = load_data_from_dynamo(nombreTabla,files)
    X=df.iloc[:,0]
    y=df.iloc[:,1]
    y=np.asarray(y)
    
    return (X,y)
    

# def testQueryTrainning():
#     file = "Ingesta10.txt"
#     df = query_trainning(file)
#     print(df)

# def testQueryTrainning2():
#     files = ["Ingesta10.txt","Ingesta12.txt"]
#     df = load_data_from_dynamo(files)
#     print(df)

# def testQueryTrainning3():
#     files = ["DB_proyecto_SA2.txt"]
#     (X,y) = carga_datos2(files)
#     print("X")
#     print(X)
#     print("y")
#     print(y)

# def testQueryTrainning4():
#     (X,y) = carga_datos(config._PATH_DATA) 
#     print("X")
#     print(X)
#     print("y")
#     print(y)
    
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



def main(X,y):
    tiempo_i = time.time() 
    file_model = f"{uuid.uuid4()}_{config._MODELO}"
    path_model = f"/tmp/{file_model}"
    bagOfWords = modeloBagOfWords(X,path_model)
    (model,Errores,Precision,Recall,F1score,tiempoEjec) = trainningNetwork(bagOfWords,y)
    file_model_mlp = f"{uuid.uuid4()}_{config._MODELO_NET}"
    path_model_net = f"/tmp/{file_model_mlp}"
    saveModel(path_model_net,model)
    lapso_tiempo = time.time() - tiempo_i
    return (Errores,Precision,Recall,F1score,lapso_tiempo,path_model,path_model_net,file_model,file_model_mlp)

def delete_model(idFile:str):
    dynamodb = boto3.resource('dynamodb')
    
    table = dynamodb.Table(config._TABLE_MODELS)
    try:
        response = table.delete_item(
            Key={
                'idFile': idFile,
            },
        )
    except ClientError as e:
        raise
    else:
        return response

def put_model(idFile:str, file_model:str, file_model_mlp:str, bucket:str):
    dynamodb = boto3.resource('dynamodb')

    table = dynamodb.Table(config._TABLE_MODELS)
    response = table.put_item(
       Item={
            'idFile': idFile,
            'info': {
                'bucket':bucket,
                'file_model': file_model,
                'file_model_mlp': file_model_mlp
            }
        }
    )
    return response

def handler(event, context):
    (nombreTabla, keys) = ("",[])
    try:
        (nombreTabla, keys) = main_handler2(event, context)
    except:
        pass

    return main_handler1(nombreTabla, keys)

def main_handler2(event, context):
    ### No borrar #######
    print(event)
    logger.info(f'Evento:{event}')
    nombreTabla = ""
    for record in event['Records']:
        print("test")
        #payload = record['body']
        payload = json.loads(record["body"])
        print(str(payload))
        for r in payload['requestPayload']['Records']:
            name = r['s3']['bucket']['name']
            key = r['s3']['object']['key']
            keys.append(key)
            print(f"El nombre del bucket es : {name}")
            print(f"El nombre de la clave es : {key}")
        nombreTabla = payload['responsePayload']['body']['tablaDynamodb']
        
    return (nombreTabla, keys)


def main_handler1(nombreTabla, keys):
    (X,y) = carga_datos2(nombreTabla,keys)
    #print(X.shape,y.shape)

    (Errores,Precision,Recall,F1score,lapso_tiempo,path_model,path_model_net,file_model,file_model_mlp) = main(X,y)
    print(f"Modelo almacenado en: {path_model_net} tiempo total: {lapso_tiempo} s")

    if upload_file(path_model, config._BUCKET_MODELS, object_name=file_model):
        print(f"Modelo {file_model} creado en {config._BUCKET_MODELS} S3") 
    else: 
        print(f"Error subiendo al modelo {file_model} a S3")

    
    if upload_file(path_model_net, config._BUCKET_MODELS, object_name=file_model_mlp):
        print(f"Modelo {file_model_mlp} creado en {config._BUCKET_MODELS} S3") 
    else: 
        print(f"Error subiendo al modelo {file_model_mlp} a S3")
    
    key = '_'.join(keys)

    delete_model(idFile=key)
    put_model(idFile=key, file_model=file_model, file_model_mlp=file_model_mlp,bucket=config._BUCKET_MODELS)
    print(f"Modelos {file_model_mlp} y {file_model} registrados en tabla {config._TABLE_MODELS}")
        
    return {
        'statusCode': 200,
        'body': {
            'path_model':path_model,
            'path_model_net':path_model_net,
            'lapso_tiempo':str(lapso_tiempo),
            'mediaPrecision':str(np.mean(Precision))
        }  
    }

if __name__ == "__main__":
    keys = ["Ingesta11.txt"]
    
    (X,y) = carga_datos2(nombreTabla=config._TABLE_INGEST,files=keys)

    (Errores,Precision,Recall,F1score,lapso_tiempo,path_model,path_model_net,file_model,file_model_mlp) = main(X,y)
    print(f"Modelos almacenado en: {path_model_net} y {path_model} tiempo total: {lapso_tiempo} s")
    
    if upload_file(path_model, config._BUCKET_MODELS, object_name=file_model):
        print(f"Modelo {file_model} creado en {config._BUCKET_MODELS} S3") 
    else: 
        print(f"Error subiendo al modelo {file_model} a S3")

    
    if upload_file(path_model_net, config._BUCKET_MODELS, object_name=file_model_mlp):
        print(f"Modelo {file_model_mlp} creado en {config._BUCKET_MODELS} S3") 
    else: 
        print(f"Error subiendo al modelo {file_model_mlp} a S3")

    key = '_'.join(keys)
    delete_model(idFile=key)
    put_model(idFile=key, file_model=file_model, file_model_mlp=file_model_mlp,bucket=config._BUCKET_MODELS)
    print(f"Modelos {file_model_mlp} y {file_model} registrados en tabla {config._TABLE_MODELS}")








    
    
    