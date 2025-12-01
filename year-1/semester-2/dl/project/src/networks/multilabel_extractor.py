import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from typing import Dict, List

class MultiLabelExtractor(nn.Module):

    def __init__(self, config: Dict[str, any]):
        """
        Initializes the MultiLabelExtractor with the given configuration. The configuration
        contains the hyperparameters of the MultiLabelExtractor, such as the input size, hidden size,
        number of layers, output size, dropout rate, etc.

        Args:
            config (Dict[str, any]): The configuration of the MultiLabelExtractor.
        """
        super(MultiLabelExtractor, self).__init__()
        self._parse_config(config) 

        self.sentence_transformer = SentenceTransformer('sentence-transformers/LaBSE')
        self.sentence_transformer = self.sentence_transformer.to(self.device)
        self.sentence_transformer.train()

        # Freeze the sentence transformer model, so it's gradients are not updated during training.
        # This is because we are using the sentence transformer model as a feature extractor.
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

        self.embeddings_dimension = self.sentence_transformer.get_sentence_embedding_dimension()

        layers = []
        for _ in range(self.hidden_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(self.embeddings_dimension, self.hidden_layer_size))
            else:
                layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            layers.append(nn.ReLU())

            if self.dropout_rate:
                layers.append(nn.Dropout(p=self.dropout_rate))

        # Add the output layer with the output dimensions equal to the number of classes.
        layers.append(nn.Linear(self.hidden_layer_size, self.output_classes))
        self.classifier = nn.Sequential(*layers)

        self.to(self.device)

    def forward(self, x: str) -> torch.Tensor:
        """
        Performs a forward pass through the MultiLabelExtractor. The forward pass consists of the following steps:
        1. Extract sentence embeddings using the sentence transformer model.
        2. Pass the embeddings through the classifier to get the output.

        Args:
            x (str): The input text.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Extract sentence embeddings using the sentence transformer model.
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode([x], convert_to_tensor=True).to(self.device)

        # Pass the embeddings through the classifier to get the output.
        output = self.classifier(embeddings)

        return output

    def _parse_config(self, config: Dict[str, any]):
        """
        Parses the configuration and initializes the corresponding attributes.

        Args:
            config (Dict[str, any]): The configuration of the MultiLabelExtractor.
        """
        self.hidden_layer_size = config['hidden_layer_size']
        self.hidden_layers = config['hidden_layers']
        self.output_classes = config['output_classes']
        self.dropout_rate = config['dropout_rate']
        self.device = config['device']