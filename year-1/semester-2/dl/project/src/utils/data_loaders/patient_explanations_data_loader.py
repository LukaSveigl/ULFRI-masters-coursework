import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

class PatientExplanationDataLoader():
    """
    The PatientExplanationDataLoader class is responsible for loading the patient explanations dataset and
    preparing it for use in the symptom extractor by generating a train-test split.
    """

    def __init__(self, dataset_path: str):
        """
        Initializes the PatientExplanationDataLoader with the path to the dataset.

        Args:
            dataset_path (str): The path to the dataset.
        """
        self.dataset_path = dataset_path
        self.label_encoder = LabelEncoder()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the patient explanations dataset and generates a train-test split.

        This function reads the patient explanations dataset from the path provided during the object initialization.
        It then generates a train-test split, with 80% of the data used for training and 20% for testing.

        Returns:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            X_test (np.ndarray): The test data.
            y_test (np.ndarray): The test labels.
        """
        X, y = [], []
        with open(self.dataset_path, 'r') as file:
            dataset = [json.loads(line) for line in file.readlines()]
            
            for data in dataset:
                X.append(data['text'])
                y.append(data['labels'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # The labels are lists of strings, such as O (outside of a named entity), B-SYMPTOM (beginning of a symptom entity),
        # I-SYMPTOM (inside of a symptom entity). We need to convert these labels to integers for training the model.
        # Flatten y_train and y_test
        y_train_flat = [label for sentence in y_train for label in sentence]
        y_test_flat = [label for sentence in y_test for label in sentence]

        # Fit and transform the flattened labels
        y_train_flat = self.label_encoder.fit_transform(y_train_flat)
        y_test_flat = self.label_encoder.transform(y_test_flat)

        # Transform the original y_train and y_test
        y_train = [self.label_encoder.transform(sentence) for sentence in y_train]
        y_test = [self.label_encoder.transform(sentence) for sentence in y_test]

        return X_train, y_train, X_test, y_test
    
    def get_vocabulary_size(self) -> int:
        """
        Returns the size of the vocabulary in the patient explanations dataset.

        This function reads the patient explanations dataset from the path provided during the object initialization.
        It then constructs a vocabulary from the text data in the dataset and returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary in the patient explanations dataset.
        """
        with open(self.dataset_path, 'r') as file:
            data = [json.loads(line) for line in file.readlines()]
            text_data = [item['text'] for item in data]

        vocabulary = set()
        for text in text_data:
            vocabulary.update(text.split())

        return len(vocabulary)
    
    def get_vocabulary(self) -> List[str]:
        """
        Returns the vocabulary of the patient explanations dataset.

        This function reads the patient explanations dataset from the path provided during the object initialization.
        It then constructs a vocabulary from the text data in the dataset and returns the vocabulary.

        Returns:
            list: The vocabulary of the patient explanations dataset.
        """
        with open(self.dataset_path, 'r') as file:
            data = [json.loads(line) for line in file.readlines()]
            text_data = [item['text'] for item in data]

        vocabulary = set()
        for text in text_data:
            vocabulary.update({word.replace('.', '').replace(',', '') for word in text.split()})

        return list(vocabulary)
    
    def get_output_classes(self) -> int:
        """
        Returns the number of output classes in the patient explanations dataset.

        Returns:
            int: The number of output classes in the patient explanations dataset.
        """
        return len(self.label_encoder.classes_)
    
    def get_output_classes_mapping(self) -> List[str]:
        """
        Returns the mapping of output classes in the patient explanations dataset.

        Returns:
            list: The mapping of output classes in the patient explanations dataset.
        """
        return list(self.label_encoder.classes_)
    
