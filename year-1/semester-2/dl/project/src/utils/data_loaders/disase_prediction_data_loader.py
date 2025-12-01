# This module contains the DiseasePredictionDataLoader class which is responsible for loading and 
# preprocessing the disease prediction datasets.
# 
# The DiseasePredictionDataLoader class reads the train and test datasets from the provided paths, 
# separates the target column (assumed to be 'prognosis'), and converts the dataframes to numpy 
# arrays which can be directly fed into a neural network.
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from typing import Tuple

class DiseasePredictionDataLoader():
    """
    A data loader for the disease prediction datasets.

    This class is responsible for loading and preprocessing the disease prediction datasets. 
    It reads the train and test datasets from the provided paths, separates the target column (assumed to be 'prognosis'), 
    and converts the dataframes to numpy arrays which can be directly fed into a neural network.

    Attributes:
        train_dataset_path (str): The path to the train dataset.
        test_dataset_path (str): The path to the test dataset.
    """

    def __init__(self, train_dataset_path: str, test_dataset_path: str):
        """
        Initialize the DiseasePredictionDataLoader with the paths to the train and test datasets.

        Args:
            train_dataset_path (str): The path to the train dataset.
            test_dataset_path (str): The path to the test dataset.
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the train and test datasets, combine the target columns, and convert the data to a form suitable for a neural network.

        This function reads the train and test datasets from the paths provided during the object initialization. 
        It then separates the target column (assumed to be 'prognosis') from the datasets. 
        Finally, it converts the dataframes to numpy arrays which can be directly fed into a neural network.

        Returns:
            X_train (numpy.ndarray): The training data.
            y_train (numpy.ndarray): The training labels.
            X_test (numpy.ndarray): The test data.
            y_test (numpy.ndarray): The test labels.
        """
        # Load the train and test datasets.
        train_df = pd.read_csv(self.train_dataset_path)
        test_df = pd.read_csv(self.test_dataset_path)

        # Separate the features and the labels.
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        self.input_feature_dimensions = X_train.shape[1]
        self.output_classes = len(np.unique(y_train))
        self.output_classes_mapping = np.unique(y_train)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test) 

        # Convert the data to PyTorch tensors.
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return X_train, y_train, X_test, y_test
    
    def get_input_feature_dimensions(self) -> int:
        """
        Get the dimensions of the input features.

        Returns:
            int: The number of input features.
        """
        return self.input_feature_dimensions
    
    def get_output_classes(self) -> int:
        """
        Get the number of output classes.

        Returns:
            int: The number of output classes.
        """
        return self.output_classes
    
    def get_output_classes_mapping(self) -> np.ndarray:
        """
        Get the mapping of the output classes.

        Returns:
            np.ndarray: The mapping of the output classes.
        """
        return self.output_classes_mapping
    