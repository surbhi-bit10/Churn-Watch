import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    one_enc=pickle.load(file)

st.title('Customer Churn PRediction')



geography = st.selectbox('Geography', one_enc.categories_[0])
gender = st.selectbox('Gender', one_enc.categories_[1])

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography':[geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

cat_sparsed=one_enc.transform(input_data[['Geography','Gender']])

cat_encoded=pd.DataFrame(cat_sparsed,columns=one_enc.get_feature_names_out())

input_data=pd.concat([input_data,cat_encoded],axis=1)

input_data=input_data.drop(['Geography','Gender'],axis=1)

pred=model.predict(input_data)

prediction_probablity=pred[0][0]
st.write(f'Churn Probability: {prediction_probablity:.2f}')

if prediction_probablity > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
