import torch
import pandas as pd
from typing import List, Tuple

class SymptomsTranscoder:
    def __init__(self, symptoms_dataset, explanations_dataset, method='default') -> None:
        """
        Initializes the SymptomsTranscoder with the paths to the symptoms dataset and the explanations dataset.

        Args:
            symptoms_dataset (str): The path to the symptoms dataset.
            explanations_dataset (str): The path to the explanations dataset.
            method (str): The method to use for transcoding the symptoms. Default is 'default'.
        """
        self.symptoms_dataset = symptoms_dataset
        self.explanations_dataset = explanations_dataset
        if method not in {'default', 'keybert', 'multilabel'}:
            raise ValueError('Invalid method. Supported methods are: default, keybert, multilabel.')
        self.method = method

    def transcode(self, text: str, symptoms: List[int], labels: List = None) -> torch.Tensor:
        """
        Wrapper function to transcode the symptoms based on the method specified during the object initialization.

        Args:
            text (str): The input text.
            symptoms (List[int]): The list of symptom labels.

        Returns:
            torch.Tensor: The transcoded symptoms tensor.
        """
        if self.method == 'default':
            return self._transcode_default(text, symptoms)
        elif self.method == 'keybert':
            return self._transcode_keybert(text, symptoms)
        elif self.method == 'multilabel':
            return self._transcode_multilabel(text, symptoms, labels)

    def _transcode_default(self, text: str, symptoms: List[int]) -> torch.Tensor:
        """
        Transcodes the symptoms from the format returned by the symptom extractor 
        (BiDirectionalLSTMExtractor) into a format required by the (FeedForwardDiseaseClassifier).

        The symptoms are in the format of a list of labels, where each label is an integer and
        represents a word being either O, B-SYMPTOM, or I-SYMPTOM. We need to convert this format
        into a format where each symptom present symptom is represented by a 1 and each missing symptom
        is represented by a 0.

        Args:
            text (str): The input text.
            symptoms (List[int]): The list of symptom labels.

        Returns:
            torch.Tensor: The symptoms tensor.
        """
        if len(text.split(' ')) != len(symptoms):
            raise ValueError('The number of symptom tags does not match the number of words in the sentence.')
        
        symptom_words = text.split(' ')
        symptom_labels = symptoms

        symptom_strings = []
        current_symptom = ''
        for i in range(len(symptom_words)):
            print(symptom_words[i], symptom_labels[i].item())
            if symptom_labels[i] == 0:
                if current_symptom:
                    symptom_strings.append(current_symptom.strip())
                    current_symptom = ''
                current_symptom += symptom_words[i] + ' '
            elif symptom_labels[i] == 1:
                current_symptom += symptom_words[i] + ' '
            else:
                if current_symptom:
                    symptom_strings.append(current_symptom.strip())
                    current_symptom = ''

        if current_symptom:
            symptom_strings.append(current_symptom.strip())

        symptom_strings = [symptom_string.replace(' ', '_') for symptom_string in symptom_strings if symptom_string != '']

        training_symptoms = pd.read_csv(self.symptoms_dataset).columns[:-1]
        symptoms_vector = []
        for symptom in training_symptoms:
            if symptom in symptom_strings:
                symptoms_vector.append(1)
            else:
                symptoms_vector.append(0)

        # Convert the symptoms vector to a tensor
        symptoms_tensor = torch.tensor(symptoms_vector)
        return symptoms_tensor
    
    def _transcode_keybert(self, text: str, symptoms: List[Tuple]) -> torch.Tensor:
        """
        Transcodes the symptoms from the format returned by the symptom extractor 
        (KeyBert) into a format required by the (FeedForwardDiseaseClassifier).

        Args:
            text (str): The input text.
            symptoms (List[int]): The list of symptom labels.

        Returns:
            torch.Tensor: The symptoms tensor.
        """
        training_symptoms = pd.read_csv(self.symptoms_dataset).columns[:-1]

        symptoms_text = [symptom.replace('_', ' ') for symptom in training_symptoms]
        symptoms_vector = [0] * len(training_symptoms)

        for symptom in symptoms:
            sym, prob = symptom

            if sym in symptoms_text:
                symptoms_vector[symptoms_text.index(sym)] = 1

        # Convert the symptoms vector to a tensor
        symptoms_tensor = torch.tensor(symptoms_vector)
        return symptoms_tensor

    def _transcode_multilabel(self, text: str, symptoms: List[int]) -> torch.Tensor:
        """
        Transcodes the symptoms from the format returned by the symptom extractor 
        (MultiLabelExtractor) into a format required by the (FeedForwardDiseaseClassifier).

        Args:
            text (str): The input text.
            symptoms (List[int]): The list of symptom labels.

        Returns:
            torch.Tensor: The symptoms tensor.
        """
        training_symptoms = pd.read_csv(self.symptoms_dataset).columns[:-1]

        symptoms_text = [symptom.replace('_', ' ') for symptom in training_symptoms]
        symptoms_vector = [0] * len(training_symptoms)

        for symptom in symptoms:
            if symptom in symptoms_text:
                symptoms_vector[symptoms_text.index(symptom)] = 1

        # Convert the symptoms vector to a tensor
        symptoms_tensor = torch.tensor(symptoms_vector)
        return symptoms_tensor