import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import streamlit as st

#===================================================================================================================================


df=pd.read_csv("Expresso_churn_dataset.csv")
df_cl=df.dropna(axis=1)
df=df.dropna(subset=['REGION','MONTANT','FREQUENCE_RECH','REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','ON_NET',

                     'ORANGE','TIGO','ZONE1','ZONE2','TOP_PACK','FREQ_TOP_PACK']).reset_index(drop=True)

#===================================================================================================================================


encoder=LabelEncoder()
df['user_id']=encoder.fit_transform(df['user_id'])
df['REGION']=encoder.fit_transform(df['REGION'])
df['TENURE']=encoder.fit_transform(df['TENURE'])
df['MRG']=encoder.fit_transform(df['MRG'])
df['TOP_PACK']=encoder.fit_transform(df['TOP_PACK'])


#===================================================================================================================================



st.title('profilage de la base de données')
st.header("Introduction")
st.write(":blue[Ce tableau de bord montre un exemple de visualisation de données avec Streamlit.]")
st.markdown("""
* EDA
* MACHIN LEARNING
    
""")
#===================================================================================================================================

if st.sidebar.button("please choice here:"):
    
    st.sidebar.button("MONTANT")
    st.sidebar.button("REVENUE")
    st.sidebar.button("DATA_VOLUME") 
    st.sidebar.button("ORANGE")
    st.sidebar.button("TIGO")
    st.sidebar.button("ZONE1")
    st.sidebar.button("ZONE2")
    st.sidebar.button("REGION")


    option = st.selectbox(
    'Choisissez une colonne à afficher',
    ('MONTANT', 'REVENUE', 'DATA_VOLUME','ORANGE','TIGO','ZONE1','ZONE2','REGION')
    )
    

st.write('REMPLIR PAR DES VALEUR')  
#===================================================================================================================================


y= df['CHURN'] 
x = df.drop('CHURN',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
df= DecisionTreeClassifier()
df.fit(x_train,y_train)


st.title('Prédiction avec Streamlit')

#===================================================================================================================================



input_values = []
for i in range(18):
    value = st.number_input(f'Input {i+1}', key=f'input_{i}')
    input_values.append(value)

input_array = np.array(input_values).reshape(1, -1)
prediction = df.predict(input_array)

#==================================================================================================================

if st.button('Prédire'):
    st.write(f'La prédiction est : {prediction[0]}')






