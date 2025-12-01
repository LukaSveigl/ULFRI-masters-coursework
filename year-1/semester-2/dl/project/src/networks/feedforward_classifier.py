# feedforward_classifier.py
# 
# This file contains the FeedForwardDiseaseClassifier class, a simple feedforward neural network model 
# used for disease classification based on input features (symptoms). The model is initialized with a 
# configuration dictionary containing hyperparameters such as the number of hidden layers, hidden layer size, 
# output classes, and dropout rate.
# 
# The FeedForwardDiseaseClassifier class is essential for the disease diagnosis system, as it provides the 
# core functionality of predicting diseases based on symptoms. It is used in the training and prediction 
# phases of the system.
import torch
import torch.nn as nn

from typing import Dict

class FeedForwardDiseaseClassifier(nn.Module):
    """
    The FeedForwardDiseaseClassifier class is a simple feedforward neural network
    that is used to classify diseases based on the input features (symptoms). 
    """

    def __init__(self, config: Dict[str, any]):
        """
        Initializes the FeedForwardDiseaseClassifier with the given configuration. The configuration
        contains the hyperparameters of the classifier, such as the number of hidden layers, the hidden layer
        size, the output classes, the dropout rate, etc.

        Args:
            config (Dict[str, any]): The configuration of the BaselineClassifier.
        """
        super(FeedForwardDiseaseClassifier, self).__init__()
        self._parse_config(config)

        # Construct the classifier according to the configuration.
        layers = []
        for i in range(self.hidden_layers):
            # The first layer should have the input dimensions equal to the embedding dimensions. All
            # other layers should have the input dimensions equal to the hidden layer size.
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.hidden_layer_size))
            else:
                layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            if self.activation_function == 'relu':
                layers.append(nn.ReLU())
            elif self.activation_function == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.activation_function == 'tanh':
                layers.append(nn.Tanh())

            if self.dropout_rate:
                layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(self.hidden_layer_size, self.output_classes))
        self.classifier = nn.Sequential(*layers)
        self.classifier = self.classifier.to(self.device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the classifier.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        x = x.to(self.device)
        x = self.classifier(x)

        return x
    
    def _parse_config(self, config: Dict[str, any]):
        """
        Parses the configuration dictionary and initializes the classifier with the specified hyperparameters.

        Args:
            config (Dict[str, any]): The configuration of the BaselineClassifier.

        Raises:
            ValueError: If any of the required hyperparameters are missing in the configuration.
        """
        if not config:
            raise ValueError("Configuration dictionary is required.")
        if 'input_size' not in config:
            raise ValueError("Input size is required in the configuration.")
        if 'hidden_layers' not in config:
            raise ValueError("Number of hidden layers is required in the configuration.")
        if 'hidden_layer_size' not in config:
            raise ValueError("Hidden layer size is required in the configuration.")
        if 'output_classes' not in config:
            raise ValueError("Number of output classes is required in the configuration.")
        if 'dropout_rate' not in config:
            raise ValueError("Dropout rate is required in the configuration.")

        self.input_size = config.get('input_size')
        self.hidden_layers = config.get('hidden_layers')
        self.hidden_layer_size = config.get('hidden_layer_size')
        self.output_classes = config.get('output_classes')
        self.dropout_rate = config.get('dropout_rate')
        self.activation_function = config.get('activation_function', 'relu')
        self.device = config.get('device', 'cpu')
        