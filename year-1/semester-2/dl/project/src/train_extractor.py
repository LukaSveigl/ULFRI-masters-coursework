import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score

from networks.bilstm_extractor import BiDirectionalLSTMExtractor
from utils.data_loaders.patient_explanations_data_loader import PatientExplanationDataLoader
from utils.classifier_utils import get_criterion, get_optimizer, get_scheduler

extractor_config = {
    'vocabulary': 100, # Is set automatically by the data loader
    'input_size': 100, # Is set automatically by the data loader
    'hidden_layers': 1,
    'hidden_layer_size': 32, # 64,
    'output_classes': 10, # Is set automatically by the data loader
    'dropout_rate': 0.8,
    'device': 'cpu' if not torch.cuda.is_available() else 'cuda'
}

training_config = {
    'epochs': 10,
    'batch_size': 32,
    'weight_decay': 0.5,

    'loss_function': 'cross_entropy',

    'optimizer': 'adam', # 'adam' or 'sgd'
    'learning_rate': 0.001,# 0.001,

    'scheduler': 'none', # 'none', 'linear', 'exponential'
    'scheduler_gamma': 0.01,
    'scheduler_step': 5
}

config = {
    'verbose': False,
    'save_model': True,
    'model_name': 'bilstm_extractor',
    'mode': 'train'  # 'train' or 'test'
}

label_map = []

def train_extractor():
    """
    Trains the patient explanation symptom extractor using the training dataset, testing the model every 5 epochs.
    """
    data_loader = PatientExplanationDataLoader(dataset_path='../data/patient_explanations/patient_explanations.json')

    X_train, y_train, X_test, y_test = data_loader.load_data()
    extractor_config['vocabulary'] = data_loader.get_vocabulary()
    extractor_config['output_classes'] = data_loader.get_output_classes()

    global label_map
    label_map = {i: label for i, label in enumerate(data_loader.get_output_classes_mapping())}

    model = BiDirectionalLSTMExtractor(config=extractor_config)
    
    criterion = get_criterion(training_config)
    optimizer = get_optimizer(training_config, model)
    scheduler = get_scheduler(training_config, optimizer)

    # Train the Bi-directional LSTM. The input is of the model is a sequence of words, and a list of labels for 
    # each word, such as O, B-SYMPTOM or I-SYMPTOM. The labels are already encoded as integers. The output of the
    # model is a sequence of predicted labels for each word in the input sequence. Train the model not on the 
    # entire dataset, but on random subsets each epoch, to avoid overfitting.
    for epoch in range(training_config['epochs']):
        for i in range(0, len(X_train), np.random.randint(1, 100, None)):
        #for i in range(0, len(X_train), 1):
            X_batch = X_train[i]
            y_batch = y_train[i]
            y_batch =  torch.from_numpy(y_batch).long().to(extractor_config['device'])

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


def test_extractor():
    """
    Tests the disease prediction classifier using the testing dataset.
    """
    data_loader = PatientExplanationDataLoader(dataset_path='../data/patient_explanations/patient_explanations.json')

    X_train, y_train, X_test, y_test = data_loader.load_data()
    extractor_config['vocabulary'] = data_loader.get_vocabulary_size()
    extractor_config['output_classes'] = data_loader.get_output_classes()

    model = BiDirectionalLSTMExtractor(config=extractor_config)
    model.load_state_dict(torch.load(f'./checkpoints/{config["model_name"]}.pth'))
    model.eval()

    test(model, X_test, y_test)


def test(extractor: BiDirectionalLSTMExtractor, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Performs testing on the given dataset using the provided extractor.

    Args:
        extractor (BiDirectionalLSTMExtractor): The extractor to use for testing.
        X_test (torch.Tensor): The test dataset.
        y_test (torch.Tensor): The test labels.
    """
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X_test), 1):# training_config['batch_size']):
            # X_batch = X_test[i:i + training_config['batch_size']]
            # y_batch = y_test[i:i + training_config['batch_size']]
            X_batch = X_test[i]
            y_batch = y_test[i]

            # X_batch = X_batch.to(extractor_config['device'])
            #y_batch = y_batch.to(extractor_config['device'])
            y_batch =  torch.from_numpy(y_batch).long().to(extractor_config['device'])

            outputs = extractor(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            if config['verbose']:
                _, predicted_labels = torch.max(outputs.cpu(), dim=-1)
                predicted_labels = [label_map[label.item()] for label in predicted_labels]
                print(predicted_labels)

            y_true.append([label_map[label.item()] for label in y_batch])
            y_pred.append([label_map[label.item()] for label in torch.max(outputs.cpu(), dim=-1)[1]])
            #print(len(y_pred[-1]))
            # print(len([y for y in y_pred[-1] if y == 'B-SYMPTOM']))
            # print(len([y for y in y_pred[-1] if y == 'I-SYMPTOM']))
            # print(len([y for y in y_pred[-1] if y == 'O']))

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f'Total test examples: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Accuracy: {100 * correct / total}%')
    print(f'Precision: {precision_score(y_true, y_pred)}')
    print(f'Recall: {recall_score(y_true, y_pred)}')
    print(f'F1 Score: {f1_score(y_true, y_pred)}')


if __name__ == '__main__':
    print('Hello, world!')
    if config['mode'] == 'train':
        train_extractor()
    else:
        test_extractor()
        