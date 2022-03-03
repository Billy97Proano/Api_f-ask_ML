from crypt import methods
from flask import Flask, request
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import pickle


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

    #Realizamos el escalado de los datos 
    ss = StandardScaler()
    df = ss.fit_transform(df)
    df = pd.DataFrame(df)
    print(df)

    #predecimos a que grupo pertenece
    modelo.fit(df)

    prediccion = map(modelo.labels_)

    return prediccion
        
    

if __name__ =="__main__":
    app.run()
