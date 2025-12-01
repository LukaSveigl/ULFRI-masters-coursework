import torch

from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

from networks.feedforward_classifier import FeedForwardDiseaseClassifier
from networks.bilstm_extractor import BiDirectionalLSTMExtractor
from networks.multilabel_extractor import MultiLabelExtractor
from utils.symptoms_transcoder import SymptomsTranscoder
from utils.data_loaders.disase_prediction_data_loader import DiseasePredictionDataLoader
from utils.data_loaders.patient_explanations_data_loader import PatientExplanationDataLoader
from utils.data_loaders.patient_explanations_multilabel_data_loader import PatientExplanationDataLoader as PatientExplanationMLDataLoader

from train_classifier import classifier_config

def default():
    from train_extractor import extractor_config

    transcoder = SymptomsTranscoder(symptoms_dataset='../data/disease_prediction/Training.csv',
                                           explanations_dataset='../data/patient_explanations/patient_explanations.json')

    disease_prediction_data_loader = DiseasePredictionDataLoader(train_dataset_path='../data/disease_prediction/Training.csv',
                                                                    test_dataset_path='../data/disease_prediction/Testing.csv')
    X_train, y_train, X_test, y_test = disease_prediction_data_loader.load_data()
    classifier_config['input_size'] = disease_prediction_data_loader.get_input_feature_dimensions()
    classifier_config['output_classes'] = disease_prediction_data_loader.get_output_classes()

    patient_explanations_data_loader = PatientExplanationDataLoader(dataset_path='../data/patient_explanations/patient_explanations.json')
    X_train, y_train, X_test, y_test = patient_explanations_data_loader.load_data()
    extractor_config['vocabulary'] = patient_explanations_data_loader.get_vocabulary()
    extractor_config['output_classes'] = patient_explanations_data_loader.get_output_classes()

    symptom_label_map = {i: label for i, label in enumerate(patient_explanations_data_loader.get_output_classes_mapping())}
    disease_label_map = {i: label for i, label in enumerate(disease_prediction_data_loader.get_output_classes_mapping())}

    extractor = BiDirectionalLSTMExtractor(config=extractor_config)
    classifier = FeedForwardDiseaseClassifier(config=classifier_config)

    extractor.load_state_dict(torch.load(f'./checkpoints/bilstm_extractor.pth'))
    classifier.load_state_dict(torch.load(f'./checkpoints/feedforward_classifier.pth'))

    extractor.eval()
    classifier.eval()

    #input_text = "I feel itching and see a skin rash."
    #input_text = "itching and skin rash have been reported by the patient."
    input_text = "The patient reports nausea, fatigue and abdominal pain. vomiting has also been reported."

    # Extract symptoms from the input text
    symptoms = extractor(input_text)
    _, symptoms = torch.max(symptoms.cpu(), dim=-1)
    predicted_labels = [symptom_label_map[label.item()] for label in symptoms]
    print(f'The predicted labels are: {predicted_labels}')

    symptoms = transcoder.transcode(input_text, symptoms)

    # Predict the disease based on the extracted symptoms
    disease = classifier(symptoms.float())

    _, disease = torch.max(disease.cpu(), dim=-1)
    disease = disease.item()
    disease = disease_label_map[disease]

    print(f'The predicted disease is: {disease}')


def keybert():
    transcoder = SymptomsTranscoder(
        symptoms_dataset='../data/disease_prediction/Training.csv',
        explanations_dataset='../data/patient_explanations/patient_explanations.json',
        method='keybert'
    )

    disease_prediction_data_loader = DiseasePredictionDataLoader(train_dataset_path='../data/disease_prediction/Training.csv',
                                                                    test_dataset_path='../data/disease_prediction/Testing.csv')
    X_train, y_train, X_test, y_test = disease_prediction_data_loader.load_data()
    classifier_config['input_size'] = disease_prediction_data_loader.get_input_feature_dimensions()
    classifier_config['output_classes'] = disease_prediction_data_loader.get_output_classes()

    disease_label_map = {i: label for i, label in enumerate(disease_prediction_data_loader.get_output_classes_mapping())}

    extractor = KeyBERT('all-MiniLM-L6-v2')
    classifier = FeedForwardDiseaseClassifier(config=classifier_config)
    classifier.load_state_dict(torch.load(f'./checkpoints/feedforward_classifier.pth'))
    classifier.eval()

    input_text = "The patient reports nausea, fatigue and abdominal pain. vomiting has also been reported."
    
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit_transform([input_text])

    symptoms = extractor.extract_keywords(input_text, vectorizer=vectorizer, use_mmr=True, diversity=0.4)
    print(f'The extracted symptoms are: {symptoms}')
    
    symptoms = transcoder.transcode(input_text, symptoms)

    # Predict the disease based on the extracted symptoms
    disease = classifier(symptoms.float())

    _, disease = torch.max(disease.cpu(), dim=-1)
    disease = disease.item()
    disease = disease_label_map[disease]

    print(f'The predicted disease is: {disease}')


def multilabel():
    from train_multilabel_extractor import extractor_config, training_config

    transcoder = SymptomsTranscoder(
        symptoms_dataset='../data/disease_prediction/Training.csv',
        explanations_dataset='../data/patient_explanations/patient_explanations_simple.json',
        method='keybert'
    )

    disease_prediction_data_loader = DiseasePredictionDataLoader(train_dataset_path='../data/disease_prediction/Training.csv',
                                                                    test_dataset_path='../data/disease_prediction/Testing.csv')
    X_train, y_train, X_test, y_test = disease_prediction_data_loader.load_data()
    classifier_config['input_size'] = disease_prediction_data_loader.get_input_feature_dimensions()
    classifier_config['output_classes'] = disease_prediction_data_loader.get_output_classes()

    patient_explanations_data_loader = PatientExplanationMLDataLoader(dataset_path='../data/patient_explanations/patient_explanations_simple.json')
    X_train, y_train, X_test, y_test = patient_explanations_data_loader.load_data()
    extractor_config['output_classes'] = patient_explanations_data_loader.get_output_classes()
    label_classes = patient_explanations_data_loader.get_output_classes_mapping()

    extractor = MultiLabelExtractor(config=extractor_config)
    classifier = FeedForwardDiseaseClassifier(config=classifier_config)

    extractor.load_state_dict(torch.load(f'./checkpoints/multilabel_extractor.pth'))
    classifier.load_state_dict(torch.load(f'./checkpoints/feedforward_classifier.pth'))

    extractor.eval()
    classifier.eval()

    #input_text = "I feel itching and see a skin rash."
    #input_text = "itching and skin rash have been reported by the patient."
    input_text = "The patient reports nausea, fatigue and abdominal pain. vomiting has also been reported."

    # Extract symptoms from the input text
    symptoms = extractor(input_text)
    probs = torch.sigmoid(symptoms)
    preds = (probs > 0.20).float()

    predicted_labels = [label_classes[i] for i, label in enumerate(preds[0]) if label == 1]

    print(f'The predicted labels are: {predicted_labels}')
    print(len(predicted_labels))


if __name__ == '__main__':
    #keybert()
    multilabel()