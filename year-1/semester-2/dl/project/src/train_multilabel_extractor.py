import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from networks.multilabel_extractor import MultiLabelExtractor
from utils.data_loaders.patient_explanations_multilabel_data_loader import PatientExplanationDataLoader
from utils.classifier_utils import get_criterion, get_optimizer, get_scheduler

extractor_config = {
    'hidden_layers': 1,
    'hidden_layer_size': 1024,#512, #32, # 64,
    'output_classes': 10, # Is set automatically by the data loader
    'dropout_rate': None, #0.1,
    'device': 'cpu' if not torch.cuda.is_available() else 'cuda'
}

training_config = {
    'epochs': 10,
    'batch_size': 32,
    'weight_decay': 0.5,

    'loss_function': 'bce_with_logits', #'cross_entropy',

    'optimizer': 'adam', # 'adam' or 'sgd'
    'learning_rate': 0.001,

    'scheduler': 'none', # 'none', 'linear', 'exponential'
    'scheduler_gamma': 0.01,
    'scheduler_step': 5,
    'threshold': 0.30#0.03
}

config = {
    'verbose': False,
    'save_model': True,
    'model_name': 'multilabel_extractor',
    'mode': 'train'  # 'train' or 'test'
}

label_map = []

def train_extractor():
    """
    Trains the patient explanation symptom extractor using the training dataset, testing the model every 5 epochs.
    """
    data_loader = PatientExplanationDataLoader(dataset_path='../data/patient_explanations/patient_explanations_simple.json')

    X_train, y_train, X_test, y_test = data_loader.load_data()
    extractor_config['output_classes'] = data_loader.get_output_classes()

    global label_map
    label_map = {i: label for i, label in enumerate(data_loader.get_output_classes_mapping())}

    model = MultiLabelExtractor(config=extractor_config)
    
    criterion = get_criterion(training_config)
    optimizer = get_optimizer(training_config, model)
    scheduler = get_scheduler(training_config, optimizer)

    for epoch in range(training_config['epochs']):
        for i in range(0, len(X_train), np.random.randint(1, 100, None)):
        #for i in range(0, len(X_train), 1):
            X_batch = X_train[i]
            y_batch = y_train[i]
            y_batch =  torch.from_numpy(y_batch).to(extractor_config['device'])

            optimizer.zero_grad()
            outputs = model(X_batch)
            y_batch = y_batch.view(outputs.shape).float()
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
    data_loader = PatientExplanationDataLoader(dataset_path='../data/patient_explanations/patient_explanations_simple.json')

    X_train, y_train, X_test, y_test = data_loader.load_data()
    extractor_config['output_classes'] = data_loader.get_output_classes()

    model = MultiLabelExtractor(config=extractor_config)
    model.load_state_dict(torch.load(f'./checkpoints/{config["model_name"]}.pth'))
    model.eval()

    test(model, X_test, y_test)


def test(extractor: MultiLabelExtractor, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Performs testing on the given dataset using the provided extractor.

    Args:
        extractor (MultiLabelExtractor): The extractor to use for testing.
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
            probs = torch.sigmoid(outputs)
            preds = (probs > training_config['threshold']).float()

            if i % 500 == 0:
                print('Top 10 probabilities:', probs.squeeze().topk(10).values.cpu().numpy())

            y_pred.append(preds.squeeze().cpu().numpy())
            y_true.append(y_batch.squeeze().cpu().numpy())

    y_true = np.array(y_true)
    print(y_true.shape)
    y_pred = np.array(y_pred)
    print(y_pred.shape)

    print(f'Precision: {precision_score(y_true, y_pred, average="samples", zero_division=0)}')
    print(f'Recall: {recall_score(y_true, y_pred, average="samples", zero_division=0)}')
    print(f'F1 Score: {f1_score(y_true, y_pred, average="samples", zero_division=0)}')


if __name__ == '__main__':
    print('Hello, world!')
    if config['mode'] == 'train':
        train_extractor()
    else:
        test_extractor()
        