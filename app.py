from crypt import methods
import json
from flask import Flask, request
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import pickle
from pandas import json_normalize

app = Flask(__name__)

@app.route('/')
def hello_world():
    print(modelo.labels_)
    return 'Hello world '




@app.route('/predict', methods=['POST', 'GET'])
def predict():

    #cargamos el modelo 
    modelo = pickle.load(open('modelo.sav', 'rb'))

    #cargar datos 
    df = request.get_json()
    print(df)
    
    df_convertido = json_normalize(df['Datos'])
    print(df_convertido.info())

    #Realizamos el escalado de los datos 
    ss = StandardScaler()
    df_predecir = ss.fit_transform(df_convertido)
    print(df_predecir)
    df= pd.DataFrame(df_predecir)
    print(df)

    #Calculamos la distancia 
    #distancia = hopkins(df, df.shape[0])
    #print(distancia)

    #predecimos a que grupo pertenece
    prediccion = modelo.predict(df_predecir)
    #print("prediccion" + prediccion)

    if(prediccion.any() == 0 ):
         return {
        'grupo': 'Motorcycle rider'
    }
    
    elif(prediccion.any() == 1 ):
         return {
        'grupo': 'Pedestrian'
    }
    
    elif(prediccion.any() == 2 ):
         return {
        'grupo': 'Driver'
    }
    
    elif(prediccion.any() == 3 ):
         return {
        'grupo': 'Pedal cyclist'
    }   
        
    

if __name__ =="__main__":
    app.run()
