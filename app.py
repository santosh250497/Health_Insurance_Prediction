# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 14:22:52 2022
@author: Santosh Palve
"""

import pandas as pd
import streamlit as st
import pickle as p


st.write(""" ### Santosh Palve """)
st.write(""" # Predicted Insurance Price """)



st.image("""bg-insurance.jpg""")




class Preprocessing_OHE():
    
    def __init__(self,new_data):
        self.new_data=new_data
    
    
    def binary(self):
        
        self.new_data["Sex"]=self.new_data["Sex"].apply(lambda x: 1 if x == "Male"  else 0)
        self.new_data["Smoker"]=self.new_data["Smoker"].apply(lambda x: 1 if x == "Yes"  else 0)
        
        return self.new_data
    
    


def input_data():
    
    age=st.slider(label="Age",min_value=18,max_value=64,step=1),
    
    sex=st.selectbox("Sex",("Male","Female")),
    
    bmi=st.number_input(label="BMI",min_value=18.0,max_value=47.0,step=0.0001),
    
    children=st.slider(label="Children",min_value=0,max_value=5,step=1),
    
    smoker=st.selectbox("Smoker",("No","Yes")),
    
    
    
    return age,sex,bmi,children,smoker


def create_dataframe():
    
    age,sex,bmi,children,smoker=input_data()
    
    features_dict={"Age":age,"Sex":sex,
                   "BMI":bmi,"Children":children,
                   "Smoker":smoker}
    
    new_data=pd.DataFrame(features_dict)
    
    return new_data



def preprocess(new_data):
    
    ohe_preproccesing=Preprocessing_OHE(new_data)

    new_data=ohe_preproccesing.binary()
    
    
    
    return new_data


def predict(new_data):
    
    p_out = open("model_rf.pkl", "rb")
    model = p.load(p_out) 
    
    return model.predict(new_data)
    
def main():
    
   
    
    new_data=create_dataframe() 
    
    st.subheader("User Input")
    st.table(new_data)
 
    

    new_data=preprocess(new_data) 
    
    

    
    
    if st.button(label='Predicted'):
        
        charges=predict(new_data)
        st.success(f'The estimated health insurance charge is: $ {charges} USD')
        
        
    
        

if __name__ == "__main__":
    main()