# train_classifier.py
# 
# This file contains a script for training and testing a FeedForwardDiseaseClassifier. The script uses PyTorch 
# and a custom data loader to load the disease prediction dataset. It also uses utility functions to get the 
# criterion (loss function), optimizer, and learning rate scheduler based on the provided configuration.
# 
# The classifier configuration includes parameters such as the input size, number of hidden layers and their size, 
# number of output classes, dropout rate, activation function, and the device to run on (CPU or GPU).
# 
# The training configuration includes parameters such as the number of epochs, batch size, weight decay, loss function, 
# optimizer, learning rate, scheduler, and scheduler gamma.
# 
# The script also has a configuration for verbosity and whether to save the trained model.
import torch

from networks.feedforward_classifier import FeedForwardDiseaseClassifier
from utils.data_loaders.disase_prediction_data_loader import DiseasePredictionDataLoader
from utils.classifier_utils import get_criterion, get_optimizer, get_scheduler

classifier_config = {
    'input_size': 100,
    'hidden_layers': 2,
    'hidden_layer_size': 64,
    'output_classes': 10,
    'dropout_rate': 0.5,
    'activation_function': 'relu', # 'relu', 'sigmoid', 'tanh'
    'device': 'cpu' if not torch.cuda.is_available() else 'cuda'
}

training_config = {
    'epochs': 10,
    'batch_size': 32,
    'weight_decay': 0.0,

    'loss_function': 'cross_entropy',

    'optimizer': 'adam', # 'adam' or 'sgd'
    'learning_rate': 0.001,

    'scheduler': 'none', # 'none', 'linear', 'exponential'
    'scheduler_gamma': 0.1,
}

config = {
    'verbose': False,
    'save_model': False,
    'model_name': 'feedforward_classifier',
    'mode': 'train'  # 'train' or 'test'
}


def train_classifier():
    """
    Trains the disease prediction classifier using the training dataset, testing the model every 5 epochs.
    """
    data_loader = DiseasePredictionDataLoader(train_dataset_path='../data/disease_prediction/Training.csv',
                                              test_dataset_path='../data/disease_prediction/Testing.csv')
    X_train, y_train, X_test, y_test = data_loader.load_data()
    classifier_config['input_size'] = data_loader.get_input_feature_dimensions()
    classifier_config['output_classes'] = data_loader.get_output_classes()

    model = FeedForwardDiseaseClassifier(config=classifier_config)
    
    criterion = get_criterion(training_config)
    optimizer = get_optimizer(training_config, model)
    scheduler = get_scheduler(training_config, optimizer)

    for epoch in range(training_config['epochs']):
        for i in range(0, len(X_train), training_config['batch_size']):
            X_batch = X_train[i:i + training_config['batch_size']]
            y_batch = y_train[i:i + training_config['batch_size']]

            X_batch = X_batch.to(classifier_config['device'])
            y_batch = y_batch.to(classifier_config['device'])

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        print(f'Epoch: {epoch + 1}/{training_config["epochs"]}, Loss: {loss.item()}')

        if epoch % 5 == 0:
            model.eval()
            test(model, X_test, y_test)
            model.train()

    model.eval()
    test(model, X_test, y_test)

    if config['save_model']:
        torch.save(model.state_dict(), f'./checkpoints/{config["model_name"]}.pth')


def test_classifier():
    """
    Tests the disease prediction classifier using the testing dataset.
    """
    data_loader = DiseasePredictionDataLoader(train_dataset_path='../data/disease_prediction/Training.csv',
                                              test_dataset_path='../data/disease_prediction/Testing.csv')
    X_train, y_train, X_test, y_test = data_loader.load_data()
    classifier_config['input_size'] = data_loader.get_input_feature_dimensions()
    classifier_config['output_classes'] = data_loader.get_output_classes()

    model = FeedForwardDiseaseClassifier(config=classifier_config)
    model.load_state_dict(torch.load(f'./checkpoints/{config["model_name"]}.pth'))
    model.eval()

    test(model, X_test, y_test)


def test(classifier: FeedForwardDiseaseClassifier, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Performs testing on the given dataset using the provided classifier.

    Args:
        classifier (FeedForwardDiseaseClassifier): The classifier to use for testing.
        X_test (torch.Tensor): The test dataset.
        y_test (torch.Tensor): The test labels.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            X_test[i] = X_test[i].to(classifier_config['device'])
            y_test[i] = y_test[i].to(classifier_config['device'])

            outputs = classifier(X_test[i])
            _, predicted = torch.max(outputs.data, 0)

            # Print prediction and actual value
            if config['verbose']:
                print(f'Predicted: {predicted}, Actual: {y_test[i]}')

            total += 1
            correct += (predicted == y_test[i]).item()

    print(f'Total test examples: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Accuracy: {100 * correct / total}%')


if __name__ == '__main__':
    print('Hello, world!')
    if config['mode'] == 'train':
        train_classifier()
    else:
        test_classifier()
        