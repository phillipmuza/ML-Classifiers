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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


#%% Upload dataset and pre-processing

os.chdir('C:\\Users\\phill\\OneDrive\\Documents\\colocalisations_glia\\coloc_analysis')

def remove_columns(feature_to_keep):
    #Upload dataframe
    df = pd.read_csv('OLD_summarised_dataset.csv', index_col = False)    

    #The bayesian classifier assumes all features in a DF are continous
    #Here I've removed all non-continous features except for the feature I'm most interested in
#Columns to keep
    labels = ['objects','mean_region_volume.mm3.', 'mean_cell_volume','total_volume.mm3.','cell_density.cells.mm3.', 'cell_coverage.100percent.']

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

list_of_dataframes = [df_genotype, df_sex, df_region, df_markers]
list_of_dataframes_with_features = [df_genotype['genotype'], df_sex['sex'], df_region['brain_region'], df_markers['markers']]
#%%Bayesian Classifier

class bayesian_classifier:
    def __init__(self, dataframe, dataframe_with_feature, number_of_folds): 
        self.dataframe = dataframe
        self.dataframe_with_feature = dataframe_with_feature
        self.number_of_folds = number_of_folds
        
    def bayes_single_instance(self):
        y_labels = self.dataframe_with_feature.tolist()

        #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe, y_labels, test_size=0.2, random_state=42)

        #Create a Naive Bayes classifer 
        classifer = GaussianNB() #Naive Bayesian
        classifer.fit(X_train, y_train) #train it
        y_pred= classifer.predict(X_test) ##Make predictions on the test set

        #Evaluate:
        accuracy = accuracy_score(y_test, y_pred) #Accuracy
        print(f"{self.dataframe_with_feature} Accuracy: {accuracy}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print('Confusion matrix:')
        print(conf_matrix)
        
    def bayes_cross_validation(self):
        # X = data_with_features and y = labels
        X,y = self.dataframe, self.dataframe_with_feature.tolist()

        # Create a Naive Bayes classifer
        classifier = GaussianNB()

        # Specify the number of folds for cross-validation
        num_folds = self.number_of_folds

        # Create a cross-validation object
        kf = KFold(n_splits=num_folds, shuffle = True, random_state = 42)

        # Perform cross-validation and get accuracy scores
        accuracy_scores = cross_val_score(classifier, X,y, cv=kf, scoring='accuracy')

        # Calculate and print the mean accuracy
        mean_accuracy = np.mean(accuracy_scores)
        print(f"Model Mean Accuracy: {mean_accuracy}")

#%% Cross-validation of Bayesian Classifer
for x,y,z in zip(list_of_dataframes, list_of_dataframes_with_features, [5] * len(list_of_dataframes)):
    bayesian_classifier_instance = bayesian_classifier(x, y, z)
    print(f"Now working on model {y}")
    bayesian_classifier_instance.bayes_cross_validation()
