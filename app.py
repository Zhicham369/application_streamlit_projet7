

from flask import Flask,render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split  
from flask import  request, url_for, redirect, render_template
import numpy as np
from sklearn import preprocessing
import pickle

app = Flask(__name__)


@app.route('/')

def hello():

    return render_template('score.html')



@app.route('/predict', methods = ['POST'])

def predict():

    ###########/model/#############

    model = pickle.load(open('model.pkl','rb'))
    
    
    
    ############/prediction###############
    int_features = [float(x) for x in request.form.values()]
    print('**************',int_features,'**********')
    final = [np.array(int_features)]
    print('********',final,'**********')
    prediction = model.predict_proba(final)
    output=prediction[0]

    return render_template('result.html',pred = output)





@app.route('/predict_streamlit/<id_client>', methods = ['GET'])

def predict_streamlit(id_client):

    #id_client = request.args.get('id_client', default=1, type=int)
    #id_client=id_client.replace(" ", "")
    print('*',id_client,'*')
    print('******  id_client= ',id_client,'*********')
    ###########/model/#############
    data_client = pd.read_csv("data_API.csv", sep='\t')

    print(data_client.dtypes)
    data_client=data_client.drop(['Unnamed: 0'], axis=1)
    model = pickle.load(open('model.pkl','rb'))
    
    ############/prediction###############
    id_client=int(id_client)
    w=data_client[data_client['SK_ID_CURR']==id_client]
    print('le type de id_client est ',type(id_client))
    print('*****',data_client['SK_ID_CURR'].values,'******')
    if id_client in data_client['SK_ID_CURR'].values:
        print('client ************'  ,id_client,' *************       excist ')
    else:
        print('client ************'  ,id_client,' *************       excist pas')

    w=w.drop('SK_ID_CURR',axis=1)
    print('******  w= ',w,'*********')
    print(w.columns)
    final = [np.array(w.values.flatten().tolist())]
    print('********** final= ',final,'**********')
    prediction = model.predict_proba(final)
    output=prediction[0]
    print('********** output= ',output,'**********')
    dict_final = {
        'prediction1' : output[0],
        'prediction2' : output[1]
        }
    return jsonify(dict_final)
    



if __name__ == "__main__":

    app.run(debug = True)










