# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:50:45 2023

@author: phill
@title: Bayesian Classifer
"""

#%% Install libraries 

import importlib, subprocess

def install_libraries(*libraries):
    for library in libraries:
        try:
            importlib.import_module(library)
            print(f"{library} is already installed.")
        except ImportError:
            print(f"Installing {library}...")
            try:
                subprocess.run(["pip", "install", library], check=True)
                print(f"{library} has been successfully installed.")
            except subprocess.CalledProcessError:
                print(f"Failed to install {library}. Please install it manually.")

#Libraries needed
install_libraries("numpy", "pandas", "matplotlib", "os", "sklearn")

#Import libraries

import os, numpy as np, pandas as pd, matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Upload dataset and pre-processing

os.chdir('C:\\Users\\phill\\OneDrive\\Documents\\machine_learning\\ML_classifiers\\datasets')

def remove_columns(feature_to_keep):
    #Upload dataframe
    df = pd.read_csv('s100b_gfap_colocalisation_pmuza.csv', index_col = False)    

    #The bayesian classifier assumes all features in a DF are continous
    #Here I've removed all non-continous features except for the feature I'm most interested in
#Columns to keep
    labels = ['objects', 'mean_region_volume.mm3.', 'mean_cell_volume',  'total_volume.mm3.', 'cell_density.cells.mm3.', 'cell_coverage.100percent.']

#Keeping genotype data and binarising it
    df_subset = df.loc[:, labels + [feature_to_keep]]
    return df_subset

#Create you dataframe with your features of interest Binarise your feature characteristics
##
df_genotype = remove_columns('genotype')
df_genotype['genotype'] = df_genotype['genotype'].replace({'WT':1, 'Dp(10)2Yey':2})

## SEX
df_sex = remove_columns('sex')
df_sex['sex'] = df_sex['sex'].replace({'male':1, 'female':2})

## BRAIN REGION
df_region = remove_columns('brain_region')    
df_region['brain_region'] = df_region['brain_region'].replace({'CA1':1, 'CA3':2, 'GCL':3, 'ML_PML':4})

## MARKERS
df_markers = remove_columns('markers')
df_markers['markers'] = df_markers['markers'].replace({'GFAP+':1, 'S100B+GFAP-':2, 'S100B+GFAP+':3})

list_of_dataframes = [df_genotype['genotype'], df_sex['sex'], df_region['brain_region'], df_markers['markers']]
#%%Bayesian Classifier

class bayesian_classifier:
    def __init__(self, dataframe_with_feature): 
        self.dataframe_with_feature = dataframe_with_feature
        
        y_labels = dataframe_with_feature.tolist()

        #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df_genotype, y_labels, test_size=0.2, random_state=42)

        #Create a Naive Bayes classifer 
        classifer = GaussianNB() #Naive Bayesian
        classifer.fit(X_train, y_train) #train it
        y_pred= classifer.predict(X_test) ##Make predictions on the test set

        #Evaluate:
        accuracy = accuracy_score(y_test, y_pred) #Accuracy
        print(f"{dataframe_with_feature} Accuracy: {accuracy}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print('Confusion matrix:')
        print(conf_matrix)
        
accurracy_scores = [bayesian_classifier(x) for x in list_of_dataframes]
