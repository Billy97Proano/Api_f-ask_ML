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


modelo = pickle.load(open('modelo.sav', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():

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

    #predecimos a que grupo pertenece
    modelo.fit(df_predecir)

    prediccion = modelo.labels_

    return prediccion
        
    

if __name__ =="__main__":
    app.run()
