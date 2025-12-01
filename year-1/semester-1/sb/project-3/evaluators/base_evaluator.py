import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseEvaluator:
    """
    The base evaluator class. This class is responsible for loading the features and labels, and evaluating the
    performance of the model. It predicts the labels in a rudimentary, brute-force manner, by computing the distance
    between the feature vector and all other feature vectors, and then choosing the label of the closest feature vector.
    """

    def __init__(self, feature_extractor: str, mode: str):
        """
        Initializes the base evaluator with the given feature extractor and mode.

        :param feature_extractor: The feature extractor to use.
        :param mode: The mode to use.
        """

        if feature_extractor not in {'resnet', 'surf', 'lbp', 'hog'}:
            raise ValueError('Invalid feature extractor')
        if mode not in {'train', 'test', 'val'}:
            raise ValueError('Invalid mode')

        self.features_folder_dir = '../datasets/ears/features-' + feature_extractor
        self.labels_folder_dir = '../datasets/ears/labels'
        self.mode = mode

    def _load_features(self):
        """
        Loads the features from the specified features folder. The features are stored in a dictionary, where the key
        is the filename and the value is the feature vector.
        """
        self.features = dict()
        for filename in os.listdir(os.path.join(self.features_folder_dir, self.mode)):
            if filename.endswith('.txt'):
                self.features[filename] = np.loadtxt(os.path.join(self.features_folder_dir, self.mode, filename),
                                                     delimiter=',')

    def _load_labels(self):
        """
        Loads the labels from the specified labels folder. The labels are stored in a dictionary, where the key
        is the filename and the value is the identity. All other data (mainly the bounding boxes) is ignored.
        """
        self.labels = dict()
        for filename in os.listdir(os.path.join(self.labels_folder_dir, self.mode)):
            if filename.endswith('.txt'):
                with open(os.path.join(self.labels_folder_dir, self.mode, filename)) as f:
                    lines = f.readlines()
                    self.labels[filename] = lines[0].split(' ')[0]

    def _find_closest_match(self, feature: np.ndarray, filename: str) -> str:
        """
        Finds the closest match for the given feature among all other features. The closest match is defined as the
        feature with the smallest Euclidean distance to the given feature. If the closest match is the same as the
        given feature, None is returned.

        :param feature: The feature to find the closest match for.
        :param filename: The filename of the feature.
        :return: The filename of the closest match, or None if the closest match is the same as the given feature.
        """
        closest_match = None
        closest_distance = float('inf')

        for other_filename, other_feature in self.features.items():
            if other_filename != filename:
                distance = np.linalg.norm(feature - other_feature)
                if distance < closest_distance:
                    closest_match = other_filename
                    closest_distance = distance

        return closest_match

    def evaluate(self) -> dict:
        """
        Evaluates the feature extractor on the given mode. This is done by iterating over all features, and for all
        features, finding the closest match among the other features. The predictions are stored, and at the end
        the accuracy score, precision score, recall score and f1 score are calculated and returned.

        :return: A dictionary containing the accuracy score, precision score, recall score and f1 score.
        """
        if not self.features or not self.labels:
            self._load_features()
            self._load_labels()

        # Find the closest match for each feature.
        predictions = dict()
        for filename, feature in self.features.items():
            predictions[filename] = self._find_closest_match(feature, filename)

        y_true = []
        y_pred = []

        # Calculate accuracy, precision, and recall.
        for filename, identity in self.labels.items():
            y_true.append(identity)
            y_pred.append(self.labels[predictions[filename]])

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
