from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from base_evaluator import BaseEvaluator


class SVCEvaluator(BaseEvaluator):
    """
    The SVC evaluator class. This class is responsible for loading the features and labels, and evaluating the
    performance of the model. It uses the SVC classifier to predict the labels. It trains the SVC classifier on the
    training set, and then predicts the labels for the test set.
    """

    def __init__(self, feature_extractor: str, mode: str):
        """
        Initializes the evaluator with the given feature extractor and mode.

        :param feature_extractor: The feature extractor to use.
        :param mode: The mode to use.
        """
        super().__init__(feature_extractor, mode)

    def _construct_lists(self) -> tuple:
        """
        Constructs the lists of features and labels, where the corresponding features and labels are at the same index.

        :return: Tuple of lists of features and labels.
        """
        features_list = []
        labels_list = []
        for filename in self.features.keys():
            features_list.append(self.features[filename])
            labels_list.append(self.labels[filename])
        return features_list, labels_list

    def evaluate(self) -> dict:
        """
        Evaluates the given feature extractor using the SVC classifier and returns the accuracy, precision, recall and
        F1 score. It disregards the mode parameter, as it is not needed for this evaluator. It trains the SVC classifier
        on the training set, and then predicts the labels for the test set.

        :return: A dictionary containing the accuracy score, precision score, recall score and f1 score.
        """

        # Load the train features and labels.
        self.mode = 'train'
        self._load_features()
        self._load_labels()

        # Construct the lists of features and labels, where the corresponding features and labels are at the same index.
        features_list, labels_list = self._construct_lists()

        # Create the SVC classifier.
        clf = SVC(kernel='linear', C=1.0, random_state=0)

        # Train the SVC classifier.
        clf.fit(features_list, labels_list)

        # Load the test features and labels.
        self.mode = 'test'
        self._load_features()
        self._load_labels()

        # Construct the lists of features and labels, where the corresponding features and labels are at the same index.
        features_list, labels_list = self._construct_lists()

        # Predict the labels for the test set.
        predicted_labels = clf.predict(features_list)

        # Compute the accuracy, precision, recall and F1 score.
        accuracy = accuracy_score(labels_list, predicted_labels)
        precision = precision_score(labels_list, predicted_labels, average='macro')
        recall = recall_score(labels_list, predicted_labels, average='macro')
        f1 = f1_score(labels_list, predicted_labels, average='macro')

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
