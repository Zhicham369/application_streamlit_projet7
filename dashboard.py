


import streamlit as st
import pandas as pd

from urllib.request import urlopen

import json
from lightgbm import LGBMClassifier
import shap
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from flask import  request, url_for, redirect, render_template
import plotly.express as px
import streamlit.components.v1 as components
import pickle




streamlit_data = pd.read_csv("data_dashboard.csv", sep='\t')
data=streamlit_data.drop(['Unnamed: 0'], axis=1)
liste_id = data['SK_ID_CURR'].tolist()

#affichage formulaire

st.title('Tableau de bord ')
st.subheader("Prédictions de score client à partir de N° client ")
form=st.form("name")
id_input = form.text_input('Veuillez saisir l\'identifiant d\'un client:', )
form.form_submit_button("Prédire")

sample_client = str(list(data[data['TARGET'] == 0].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_client = 'Exemples d\'id de clients  : ' +sample_client



if id_input == '': #lorsque rien n'a été saisi


    st.write(chaine_client)


elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API


    
    API_url = "https://hziate.pythonanywhere.com/predict_streamlit/"+id_input

    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        classe_predite1 = API_data['prediction1']
        classe_predite2 = API_data['prediction2']
        #affichage de la prédiction

        chaine = 'Prédiction : [ ' + str(round(classe_predite1*100)) + '%   crédit accepté'     + ' , ' + str(round(classe_predite2*100)) + '%   crédit non accepté'    + ']'

    st.write(chaine)

    feature_cols = ['PAYMENT_RATE','EXT_SOURCE_1',
    'EXT_SOURCE_2','EXT_SOURCE_3','AMT_GOODS_PRICE','AMT_ANNUITY',
    'NAME_EDUCATION_TYPE_Highereducation','AMT_REQ_CREDIT_BUREAU_QRT','CODE_GENDER']

    X = data[feature_cols]
    y = data['TARGET']


    model1 = pickle.load(open('model1.pkl','rb'))
    explainer = shap.TreeExplainer(model1)
    shap_values = explainer.shap_values(X)

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st.header('Importance globale des variables')
    fig, ax = plt.subplots(figsize=(15,5))
    ax=shap.summary_plot(shap_values, X)
    st.pyplot(fig)

    st.header('Importance locale de la première variable importante class 1 ')
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X.iloc[0, :]))
    st.header('Importance locale de la première variable importante class 0')
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X.iloc[0, :]))
 
    st.header('Distribution variable selon les classes et le positionnement de la valeur du client')
    option= 'CODE_GENDER'
    option = st.selectbox(
    'Choisi feature',
    ('CODE_GENDER', 'NAME_EDUCATION_TYPE_Highereducation'))
    fig2 = px.histogram(data_frame=data, x=option)
    st.write(fig2)   


   
    new_columns=[]
    for row in data.iterrows():
        if row[1]["SK_ID_CURR"] == int(id_input): 
            new_columns.append("client   " +str(int(id_input)))   
        else: 
            new_columns.append(row[1]["TARGET"])
    data['class']= new_columns
    st.header('Box plot comparaison entre clients ')
    box_fig = px.box(data,x="TARGET",y="EXT_SOURCE_3",color="class",points="all")
    st.write(box_fig)



st.sidebar.header('Information de client')
def user_input_features():
    PAYMENT_RATE = st.sidebar.slider('PAYMENT_RATE', value=0.13)
    EXT_SOURCE_1 = st.sidebar.slider('EXT_SOURCE_1',value=0.97)
    EXT_SOURCE_2 = st.sidebar.slider('EXT_SOURCE_2',value=0.86)
    EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3',value=0.90)
    AMT_GOODS_PRICE =st.sidebar.slider('AMT_GOODS_PRICE',value=300000.0)
    AMT_ANNUITY=st.sidebar.slider('AMT_ANNUITY',value=260000.0)
    NAME_EDUCATION_TYPE_Highereducation = st.sidebar.slider('NAME_EDUCATION_TYPE_Highereducation',value=1.0)
    AMT_REQ_CREDIT_BUREAU_QRT = st.sidebar.slider('AMT_REQ_CREDIT_BUREAU_QRT',value=261.0)
    CODE_GENDER= st.sidebar.slider('CODE_GENDER',value=1.0)
    
    data1 = {'PAYMENT_RATE': PAYMENT_RATE,
            'EXT_SOURCE_1': EXT_SOURCE_1,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3,
            'AMT_GOODS_PRICE':AMT_GOODS_PRICE,
            'AMT_ANNUITY':AMT_ANNUITY,
            'NAME_EDUCATION_TYPE_Highereducation': NAME_EDUCATION_TYPE_Highereducation,
            'AMT_REQ_CREDIT_BUREAU_QRT': AMT_REQ_CREDIT_BUREAU_QRT,
            'CODE_GENDER': CODE_GENDER}
    features = pd.DataFrame(data1, index=[0])
    return features

df = user_input_features()

model = pickle.load(open('model.pkl','rb'))
prediction = model.predict_proba(df)
output=prediction[0]

st.header('Paramètres entrée informations client pour prédire le score client ')
st.write(df)
st.write('---')

st.header('Prediction score de client à partir informations client')
st.write(output)
st.write('---')


      












