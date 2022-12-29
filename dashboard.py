


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


streamlit_data = pd.read_csv("data_dashboard.csv", sep='\t')
data=streamlit_data.drop(['Unnamed: 0'], axis=1)
liste_id = data['SK_ID_CURR'].tolist()

#affichage formulaire

st.title('Tableau de bord ')
st.subheader("Prédictions de score client à partir de N° client ")
form=st.form("name")
id_input = form.text_input('Veuillez saisir l\'identifiant d\'un client:', )
form.form_submit_button("Prédire")

sample_en_regle = str(list(data[data['TARGET'] == 0].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(data[data['TARGET'] == 1].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut


if id_input == '': #lorsque rien n'a été saisi

    st.write(chaine_en_defaut)

    st.write(chaine_en_regle)


elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API


    st.write('*',id_input,'*')
    API_url = "https://hziate.pythonanywhere.com/predict_streamlit/"+id_input

    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        classe_predite1 = API_data['prediction1']
        classe_predite2 = API_data['prediction2']
        #affichage de la prédiction

        chaine = 'Prédiction : [ ' + str(round(classe_predite1*100)) + '%   crédit accpté'     + ' , ' + str(round(classe_predite2*100)) + '%   crédit non accpté'    + ']'

    st.write(chaine)



feature_cols = ['PAYMENT_RATE','EXT_SOURCE_1',
         'EXT_SOURCE_2','EXT_SOURCE_3','AMT_GOODS_PRICE','AMT_ANNUITY',
         'NAME_EDUCATION_TYPE_Highereducation','AMT_REQ_CREDIT_BUREAU_QRT','CODE_GENDER']

X = data[feature_cols]
y = data['TARGET']


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

st.header('Paramètres entrée informations client pour prédire le score client ')
st.write(df)
st.write('---')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)  # 80% training and 20% test
# Create LGBMClassifier object
clf = LGBMClassifier(learning_rate= 0.1, max_depth= 6, min_split_gain= 0.03,
                              n_estimators= 100, num_leaves= 33,class_weight = None )
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_train)
clf = clf.fit(X_train, y_transformed)
model=clf
prediction = model.predict_proba(df)
output=prediction[0]

st.header('Prediction score de client à partir informations client')
st.write(output)
st.write('---')


model1=LGBMClassifier(learning_rate= 0.1, max_depth= 6, min_split_gain= 0.03,
                              n_estimators= 100, num_leaves= 33,class_weight = None )
model1.fit(X,y)

explainer = shap.TreeExplainer(model1)
shap_values = explainer.shap_values(X)

st.header('importance globale des variables')
fig, ax = plt.subplots(figsize=(15,5))
ax=shap.summary_plot(shap_values, X)
st.pyplot(fig)







