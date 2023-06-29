# -*- coding: utf-8 -*-
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import sklearn

import operator
from collections import Counter
import math
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
import streamlit as st

import sys

# sys.path.insert(1, "d:/bluethinq/django/lib/site-packages/streamlit_option_menu")

from streamlit_option_menu import option_menu
import base64
import numpy as np
# import tensorflow as tf


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# loading the saved models

diabetes_model = pickle.load(open('Model\diabetes_model (1).sav', 'rb'))

Heart_model = pickle.load(open("Model\heart_disease_model.sav",'rb'))

Parkinson_disease_model=pickle.load(open('Model\Parkinson_model.sav','rb'))

breast_cancer_model=pickle.load(open(r'Model\breast_cancer_model.sav','rb'))

df= pd.read_csv("Datasets\checker.csv")


# Heart_model=pickle.load(open('heart_disease_model.sav','rb'))

with st.sidebar:
    
    selected = option_menu('Diagnosis360',
                          
                          ['Symptom Checker','Diabetes Prediction',
                           'Heart Disease Prediction',"Parkinson's Disease Prediction",'Breast Cancer Prediction'],
                          icons=['journal-medical','activity','heart','house','list-task'],
                          default_index=0,
                        )
    
    # selected = st.radio(' ', ['Symptom Checker','Diabetes Prediction',
    #                        'Heart Disease Prediction',"Parkinson's Disease Prediction",'Breast Cancer Prediction'])

if selected == 'Symptom Checker':
    symptoms=[]
    for i in range(1,18):
        for symptom in df['Symptom_'+ str(i)].unique():
            symptoms.append(symptom)
    symptoms=set(symptoms)


    #The symptoms are added to a set to get a list of all the differnt symptoms present.
    symptoms=list(symptoms)
    # print(symptoms)
    symptoms=symptoms[1:]
    # print(symptoms)
    symptoms=[x for x in symptoms if type(x)==str]
    symptoms.sort()
    # symptoms = [symptom.title() for symptom in symptoms]


    # Define the input list
    inp = [0 for i in range(len(symptoms))]

    # Divide the symptoms into three columns
    col1, col2, col3 = st.columns(3)

    for i in range(len(symptoms)):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        if col.checkbox(symptoms[i]):
            inp[i] = 1

    # The above part of the code is used to get a list of the symptoms from the user which he/she is experiencing.
    #   Streamlit is being used to the get the data from the checkboxes.
    if st.button("Diagnose"):
         #As soom as the button is pressed a boolean vector is generated to be matched in the dataset.
        X = []
        for i in range(len(df)):
            symptomsbin = [0 for k in range(len(symptoms))]

            # symptomsbin = [0 for k in range()]
            for element in df.iloc[i]:
                for j in range(len(symptoms)):
                    if element == symptoms[j]:
                        symptomsbin[j] = 1
            X.append(symptomsbin)
        Y = list(df["Disease"])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)
        #have used the sklearn library to just split the data. We have implemented the decision tree on our own !

        X1 = copy.deepcopy(X_train)
        Data = []
        for i in range(len(X1)):
            X1[i].append(Y_train[i])
            Data.append(X1[i])



        # Mathematical definition of entropy
        def entropy(data):
            frequency = Counter([item[-1] for item in data])

            def item_entropy(category):
                ratio = float(category) / len(data)
                return -1 * ratio * math.log(ratio, 2)

            return sum(item_entropy(c) for c in frequency.values())

            #The function gets the best feature for split. e(v) stores the entropy times proportion of one of the sides of the split by the feature f.
            #This is done for each feature and the corresponding information gain is calculated and stored.
            #The feature which gives the highest information gain is used to split the data.
        def best_feature_for_split(data):
            baseline = entropy(data)

            def feature_entropy(f):
                def e(v):
                    partitioned_data = [d for d in data if d[f] == v]
                    proportion = (float(len(partitioned_data)) / float(len(data)))
                    return proportion * entropy(partitioned_data)

                return sum(e(v) for v in set([d[f] for d in data]))

            features = len(data[0]) - 1
            information_gain = [baseline - feature_entropy(f) for f in range(features)]
            best_feature, best_gain = max(enumerate(information_gain), key=operator.itemgetter(1))
            return best_feature


        def potential_leaf_node(data):
            count = Counter([i[-1] for i in data])
            return count.most_common(1)[0]


        def classify(tree, label, data):
            root = list(tree.keys())[0]
            node = tree[root]
            index = label.index(root)
            for k in node.keys():
                if data[index] == k:
                    if isinstance(node[k], dict):
                        return classify(node[k], label, data)
                    else:
                        return node[k]


        def create_tree(data, label):
            category, count = potential_leaf_node(data)
            if count == len(data):
                return category
            node = {}
            feature = best_feature_for_split(data)
            feature_label = label[feature]
            node[feature_label] = {}
            classes = set([d[feature] for d in data])
            for c in classes:
                partitioned_data = [d for d in data if d[feature] == c]
                node[feature_label][c] = create_tree(partitioned_data, label)
            return node

        # From the above functions, a tree is created.
        #Now, we use the classify function to get the prediction.
        #Here we have just calculated the accuracy but one can make modifications to the code to get other metrics as well.
        symptoms.append("Disease")
        tree = create_tree(Data, symptoms)
        print(tree)
        score = 0
        for i in range(len(X_test)):
            if (Y_test[i] == classify(tree, symptoms, X_test[i])):
                score += 1
            # print("label: ", Y_test[i], "predicted: ", classify(tree, symptoms, X_test[i]))
        print("accuracy= ", score / len(Y_test))
        print(classify(tree, symptoms, inp))
        # st.write("working on it.....")
        # st.write("wait some time.....")
        st.write("You are diagnosed with : ",classify(tree, symptoms, inp))
        st.write("Please consult a doctor")



if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')

    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)

    
    

# Parkinson's Prediction Page
if (selected == "Parkinson's Disease Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input("MDVP:Flo(Hz)")
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        MDVP_Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
        
    with col2:
        Shimmer_APQ5 = st.text_input("Shimmer:APQ5")
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        Shimmer_DDA = st.text_input("Shimmer:DDA")
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = Parkinson_disease_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,MDVP_Shimmer,Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
        
    


if (selected == 'Breast Cancer Prediction'):
    
    # page title
    st.title('Breast Cancer Prediction using ML')

    # Divide the input fields into three columns
    col1, col2, col3 = st.columns(3)
    # Add text input fields for the first set of features
    with col1:
        radius_mean = st.number_input('mean radius', step=1)
        texture_mean = st.number_input('mean texture', step=1)
        perimeter_mean = st.number_input('mean perimeter', step=1)
        area_mean = st.number_input('mean area', step=1)
        smoothness_mean = st.number_input('mean smoothness', step=1)
        compactness_mean = st.number_input('mean compactness', step=1)

    with col2:
        concavity_mean = st.number_input('mean concavity', step=1)
        concave_points_mean = st.number_input('mean concave points', step=1)
        symmetry_mean = st.number_input('mean symmetry', step=1)
        fractal_dimension_mean = st.number_input('mean fractal dimension', step=1)
        radius_error = st.number_input('radius error', step=1)
        texture_error = st.number_input('texture error', step=1)

    with col3:
        perimeter_error = st.number_input('perimeter error', step=1)
        area_error = st.number_input('area error', step=1)
        smoothness_error = st.number_input('smoothness error', step=1)
        compactness_error = st.number_input('compactness error', step=1)
        concavity_error = st.number_input('concavity error', step=1)
        concave_points_error = st.number_input('concave points error', step=1)

    # Add text input fields for the second set of features
    col4, col5, col6 = st.columns(3)

    with col4:
        symmetry_error = st.number_input('symmetry error', step=1)
        fractal_dimension_error = st.number_input('fractal dimension error', step=1)
        worst_radius = st.number_input('worst radius', step=1)
        worst_texture = st.number_input('worst texture', step=1)

    with col5:
        worst_smoothness = st.number_input('worst smoothness', step=1)
        worst_compactness = st.number_input('worst compactness', step=1)
        worst_concavity = st.number_input('worst concavity', step=1)
        worst_concave_points = st.number_input('worst concave points', step=1)

    # # Add text input field for the last feature
    with col6:
        # target = st.text_input('Target: 0 for benign, 1 for malignant')
        worst_symmetry = st.number_input('worst symmetry', step=1)
        worst_fractal_dimension = st.number_input('worst fractal dimension', step=1)
        worst_perimeter = st.number_input('worst perimeter', step=1)
        worst_area = st.number_input('worst area', step=1)
        
        # code for prediction
    cancer_diagnosis = ''

        # create a button for prediction
    if st.button('Breast Cancer Test Result'):
        # st.write(type(breast_cancer_model))
        cancer_prediction = breast_cancer_model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]])
            
        if (cancer_prediction[0] == 1):
            cancer_diagnosis = 'The tumor is malignant (cancerous)'
        else:
            cancer_diagnosis = 'The tumor is benign (not cancerous)'
            
    st.success(cancer_diagnosis)




# Heart Disease Prediction Page
if(selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', step=1)
        
    with col2:
        sex = st.number_input('Sex: 0-Female 1-Male', step=1)
        
    with col3:
        cp = st.number_input('"Chest Pain Type", options=[1, 2, 3, 4]', step=1)
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', step=1)
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', step=1)
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar 0: <= 120 mg/dl, 1: > 120 mg/dl', step=1)
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results (0-2)', step=1)
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', step=1)
        
    with col3:
        exang = st.number_input('Exercise Induced Angina 0: no, 1: yes', step=1)
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', step=1)
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment (1-3)', step=1)
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy(0-3)', step=1)
        
    with col1:
        thal = st.number_input('thal: 0=normal 1=fixed defect 2=reversable defect', step=1)
    
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    #create a button for prediction
    if st.button('Heart Disease Test Result'):
        # st.write(type(Heart_model))
        prediction = Heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # display the prediction
        if prediction[0] == 1:
            heart_diagnosis ="Sorry, you have a heart disease."
        else:
            heart_diagnosis ="Congratulations, you don't have a heart disease."

    st.success(heart_diagnosis)
