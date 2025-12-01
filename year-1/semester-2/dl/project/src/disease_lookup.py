# disease_lookup.py
# 
# This file contains a script that uses a dataset of diseases and their symptoms to find potential diseases
# that match a given set of symptoms. The script reads in a training and testing dataset, combines them, and
# then filters the combined dataset for diseases that have all the given symptoms.
import pandas as pd

if __name__ == '__main__':
    test_dataset = pd.read_csv('../data/disease_prediction/Testing.csv')
    train_dataset = pd.read_csv('../data/disease_prediction/Training.csv')

    combined_dataset = pd.concat([train_dataset, test_dataset])

    #symptoms = ['itching', 'skin_rash']
    # Symptoms that match Typhoid Fever
    symptoms = ['nausea', 'fatigue', 'abdominal_pain', 'vomiting']

    # Get all prognoses that have the given symptoms.
    diseases = combined_dataset[(combined_dataset[symptoms] == 1).all(axis=1)]['prognosis'].unique()

    print(f'Diseases that match the given symptoms: {diseases}')
    