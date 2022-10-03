import pandas as pd
from flask import Flask, request, Response
import pickle
import json
import os

from rossmann.Rossmann import Rossmann

#Loading model

model = pickle.load( open('model/model_rossmann.pkl', 'rb'))

# initialize API
app = Flask (__name__)

@app.route( '/rossmann/predict', methods=['POST'])

def rossmann_predict():
    test_json = request.get_json()

#Conversão do json em Data Frame
    if test_json: #there is data

        if isinstance(test_json, dict): #unico exemplo
            test_raw = pd.DataFrame(test_json, index=0)
        else: #Multiplos exemplos
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys() )

#Instanciar a Rossmann class

        pipeline = Rossmann()

        #data Cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        #Feature Engineering
        df2 = pipeline.feature_engineering(df1)
        
        #Data preparition
        df3 = pipeline.data_preparation(df2)
        
        #predictiom
        df_response =pipeline.get_prediction( model, test_raw, df3)

        return df_response

    else:
        return Response( '{}', status=200, mimetype = 'application/json')



if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host= '0.0.0.0', port = port)
