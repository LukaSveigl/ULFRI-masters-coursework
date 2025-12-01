import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
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
        self.label_encoder = MultiLabelBinarizer()

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

        # The labels are lists of keywords. We need to convert these labels to binary format for training the model.
        self.label_encoder.fit(y_train + y_test)
        y_train = self.label_encoder.transform(y_train)
        y_test = self.label_encoder.transform(y_test)

        return X_train, y_train, X_test, y_test
    
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
    
