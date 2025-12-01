import torch
import torch.nn as nn

from typing import Dict, List

class BiDirectionalLSTMExtractor(nn.Module):

    def __init__(self, config: Dict[str, any]):
        """
        Initializes the BiDirectionalLSTMExtractor with the given configuration. The configuration
        contains the hyperparameters of the BiDirectionalLSTMExtractor, such as the input size, hidden size,
        number of layers, output size, dropout rate, etc.

        Args:
            config (Dict[str, any]): The configuration of the BiDirectionalLSTMExtractor.
        """
        super(BiDirectionalLSTMExtractor, self).__init__()
        self._parse_config(config) 

        self.word_to_ix = self._build_vocab(config['vocabulary'])
        self.embedding = nn.Embedding(num_embeddings=len(self.word_to_ix), 
                                      embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_layer_size,
                            num_layers=self.hidden_layers,
                            bidirectional=True,
                            dropout=self.dropout_rate if self.hidden_layers > 1 else 0,
                            batch_first=True)

        self.fc = nn.Linear(self.hidden_layer_size * 2, len(self.word_to_ix))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.to(self.device)

    def forward(self, x: str) -> torch.Tensor:
        """
        Performs a forward pass through the BiDirectionalLSTMExtractor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x = x.split()  # tokenize the input string
        # x = [self.word_to_ix[word] for word in x]  # convert words to indices
        # x = torch.tensor(x)  # convert list to tensor
        # x = self.embedding(x)
        # lstm_out, _ = self.lstm(x.unsqueeze(0))  # add an extra dimension for batch size
        # lstm_out = self.dropout(lstm_out)
        # lstm_out = self.fc(lstm_out)
        #lstm_out = self.softmax(lstm_out)

        x = x.split(' ')  # tokenize the input string
        x = [self.word_to_ix[word.replace('.', '').replace(',', '')] for word in x]  # convert words to indices
        x = torch.tensor(x).to(self.device)  # convert list to tensor
        x = self.embedding(x).to(self.device)
        lstm_out, _ = self.lstm(x.view(1, -1, self.embedding_dim))  # reshape x to have shape (batch_size, seq_len, input_size)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.view(-1, self.hidden_layer_size * 2)  # reshape lstm_out to have shape (seq_len, hidden_layer_size * 2)
        lstm_out = self.fc(lstm_out)  # apply the fully connected layer to each word
        lstm_out = self.softmax(lstm_out)
        lstm_out = lstm_out.to(self.device)

        return lstm_out

    def _parse_config(self, config: Dict[str, any]):
        """
        Parses the configuration and initializes the corresponding attributes.

        Args:
            config (Dict[str, any]): The configuration of the BiDirectionalLSTMExtractor.
        """
        self.embedding_dim = config['input_size']
        self.hidden_layer_size = config['hidden_layer_size']
        self.hidden_layers = config['hidden_layers']
        self.output_classes = config['output_classes']
        self.dropout_rate = config['dropout_rate']
        self.device = config['device']

    def _build_vocab(self, vocab: List[str]) -> Dict[str, int]:
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        return word_to_ix