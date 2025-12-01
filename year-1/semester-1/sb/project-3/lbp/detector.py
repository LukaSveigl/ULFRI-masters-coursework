# This file contains the implementation of the Viola Jones detector, which is used to detect 
# ears in images. This file also serves as the driver file for the evaluation of the detector.

import common, utils
import json, cv2, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

class VJDetector:
    def __init__(self, classifier_path: str, class_args: dict, det_args: dict) -> object:
        """
        Constructs a VJDetector object.

        :param classifier: Path to the classifier.
        :param class_args: Arguments for the classifier.
        :param det_args: Arguments for the detector.
        """
        self.classifier = cv2.CascadeClassifier(classifier_path)
        self.class_args = class_args
        self.det_args = det_args

    def detect(self, src_path: str) -> list:
        """
        Detects objects in the image.

        :param src_path: Path to the input image.
        :returns: List of detected objects.
        """

        self.image = cv2.imread(src_path)

        # Convert the image to grayscale if necessary.
        if self.det_args['grayscale']:
            gs_image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)
        else:
            gs_image = self.image

        # Detect objects in the image.
        return self.classifier.detectMultiScale(
            image=gs_image, 
            scaleFactor=self.class_args['scale_factor'], 
            minNeighbors=self.class_args['min_neighbors'], 
            flags=self.class_args['flags'], 
            minSize=self.class_args['min_size'], 
            maxSize=self.class_args['max_size']
        )

    def visualize(self, src_path: str, out_path: str, out_path_cropped: str) -> list:
        """
        Visualizes the detected objects in the image and returns the detections.

        :param src_path: Path to the input image.
        :param out_path: Path to the output image.
        :param out_path_cropped: Path to the output image with the cropped object.
        :returns: List of detections.
        """
        self.image = cv2.imread(src_path)
        self.detections = self.detect(src_path)

        # Draw the rectangle around the detected object and save it.
        for x, y, w, h in self.detections:
            # Crop the image to the detected object and save it.
            crop_img = self.image[y:y+h, x:x+w]
            cv2.imwrite(out_path_cropped, crop_img)
            cv2.rectangle(self.image, (x,y), (x+w, y+h), (128, 255, 0), 4)
        cv2.imwrite(out_path, self.image)
        return self.detections
    

def load_ground_truths(annotations: dict, normalize: bool) -> dict:
    """
    Loads the ground truths from the annotations.

    :param annotations: Annotations from the JSON file.
    :param normalize: Whether to normalize the ground truths to the yolo format.
    :returns: Dictionary of ground truths.
    """
    print('Loading ground truths with normalize={}'.format(normalize))

    ground_truths = {}
    for key in annotations.keys():
        # Load the truth from the file and skip the class label.
        truth_file = common.SRC_IMAGES + key.replace('png', 'txt')
        ground_truths[key] = [float(x) for x in open(truth_file).readline().split()][1::]

        if normalize:
            image = cv2.imread(common.SRC_IMAGES + key)
            ground_truths[key] = utils.yolo_to_opencv(ground_truths[key][0], ground_truths[key][1], ground_truths[key][2], ground_truths[key][3], image.shape[1], image.shape[0])
    return ground_truths


def generate_args() -> tuple:
    """
    Generates the arguments for the classifier and detector.

    :returns: Tuple of arguments for the classifier and detector.
    """
    print('Generating arguments for the detector...')

    class_args = {
        'scale_factor': 1.01,
        'min_neighbors': 3,
        'flags': 0,
        'min_size': (30, 30),
        'max_size': (3000, 3000)
    }

    det_args = {
        'grayscale': False
    }

    return class_args, det_args


def evaluate(
        annotations: dict, 
        left_detector: VJDetector, 
        right_detector: VJDetector, 
        ground_truths: dict, 
        threshold: float, 
        visualize: bool, 
        file_names: pd.Series = None
    ) -> dict:
    """
    Evaluates the detector with the given parameters.

    :param annotations: Annotations from the JSON file.
    :param left_detector: Detector for the left ear.
    :param right_detector: Detector for the right ear.
    :param ground_truths: Ground truths for the images.
    :param threshold: Threshold for the IoU.
    :param visualize: Whether to visualize the results.
    :param file_names: List of file names to evaluate.
    :returns: Dictionary of results.
    """
    print('Evaluating the detector...')

    score_sum = 0
    all_results = {'TP': 0, 'FP': 0, 'FN': 0}

    if file_names is not None:
        annotations = {k: v for k, v in annotations.items() if k in set(file_names)}
        ground_truths = {k: v for k, v in ground_truths.items() if k in set(file_names)}
        print('Hello world!', len(file_names))

    # Loop through all images and try to detect ears.
    for image, ear in annotations.items():
        detections = []
        if ear == 'l':
            if visualize:
                computed_path = common.OUT_IMAGES_COMPUTED + image + '.vj.detected.png'
                computed_cropped_path = common.OUT_IMAGES_CR_COMPUTED + image + '.vj.detected.cropped.png'
                detections = left_detector.visualize(common.SRC_IMAGES + image, computed_path, computed_cropped_path)
            else:
                detections = left_detector.detect(common.SRC_IMAGES + image)
        elif ear == 'r':
            if visualize:
                computed_path = common.OUT_IMAGES_COMPUTED + image + '.vj.detected.png'
                computed_cropped_path = common.OUT_IMAGES_CR_COMPUTED + image + '.vj.detected.cropped.png'
                detections = right_detector.visualize(common.SRC_IMAGES + image, computed_path, computed_cropped_path)
            else:
                detections = right_detector.detect(common.SRC_IMAGES + image)
        else:
            raise ValueError('Invalid ear value.')

        # Calculate the score of the image.
        image_results, image_score = utils.image_score(detections, ground_truths[image], threshold)
        score_sum += image_score
        all_results['TP'] += image_results['TP']
        all_results['FP'] += image_results['FP']
        all_results['FN'] += image_results['FN']

    score = score_sum / len(annotations)
    precision = all_results['TP'] / (all_results['TP'] + all_results['FP'])
    if all_results['TP'] + all_results['FN'] == 0:
        recall = 0
    else:
        recall = all_results['TP'] / (all_results['TP'] + all_results['FN'])

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'score': score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': all_results['TP'],
        'FP': all_results['FP'],
        'FN': all_results['FN']
    }


def optimize(annotations: dict, ground_truths: dict, train_test: tuple = None) -> dict:
    """
    Optimizes the parameters of the detector by looping through all possible combinations.

    :param annotations: Annotations from the JSON file.
    :param ground_truths: Ground truths for the images.
    :returns: List of dictionaries of results and optimized parameters.
    """
    print('Optimizing the detector...')
    
    # Generate the arguments for the classifier and detector.
    class_args, det_args = generate_args()
    class_args['min_neighbors'] = 1
    class_args['scale_factor'] = 1.01

    best_f1 = 0
    best_params = {
        'min_neighbors': 0,
        'scale_factor': 0,
        'threshold': 0
    }

    results_dict = dict()

    file_names, _, _, _ = train_test

    # Loop through all possible combinations of parameters.
    for curr_min_neighbors in np.arange(class_args['min_neighbors'], 6):
        for curr_scale_factor in np.arange(class_args['scale_factor'], 1.5, 0.01):
            for curr_threshold in np.arange(0.1, 1.0, 0.1):
                class_args['min_neighbors'] = curr_min_neighbors
                class_args['scale_factor'] = curr_scale_factor

                # Create the detectors from the current combination of parameters.
                left_detector = VJDetector(common.INTEGRAL_LEFT, class_args, det_args)
                right_detector = VJDetector(common.INTEGRAL_RIGHT, class_args, det_args)

                results = evaluate(annotations, left_detector, right_detector, ground_truths, curr_threshold, False, file_names)
                results_dict[(curr_min_neighbors, curr_scale_factor, curr_threshold)] = results

                if results['f1'] > best_f1:
                    best_f1 = results['f1']
                    best_params['min_neighbors'] = curr_min_neighbors
                    best_params['scale_factor'] = curr_scale_factor
                    best_params['threshold'] = curr_threshold

    return results_dict, best_params
    

def main(eval_once: bool = True, visualize: bool = True, normalize: bool = True, train_test: tuple = None):
    """
    Main function.

    :param eval_once: Whether to evaluate the detector once or optimize it.
    :param visualize: Whether to visualize the results.
    :param normalize: Whether to normalize the ground truths to the OpenCV format.
    :param train_test: The train_test split to use.
    """
    annotations = json.load(open(common.ANNOTATIONS))
    ground_truths = load_ground_truths(annotations, normalize)

    if eval_once is True:
        if train_test is None:
            identities = pd.read_csv(common.IDENTITIES, sep=' ', header=None)
            identities.columns = ['image', 'id']

            x = identities['image']
            y = identities['id']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
            file_names = x_train
        else:
            file_names = train_test[0]
            print(file_names)
        
        # Set the best arguments calculated by the `optimize` function.
        classifier_args, detector_args = generate_args()
        classifier_args['scale_factor'] = 1.07
        classifier_args['min_neighbors'] = 1
        THRESHOLD = 0.1

        print(classifier_args, detector_args)

        results = evaluate(
            annotations, 
            VJDetector(common.INTEGRAL_LEFT, classifier_args, detector_args),
            VJDetector(common.INTEGRAL_RIGHT, classifier_args, detector_args),
            ground_truths,
            THRESHOLD,
            visualize,
            file_names
        )
        print('Score: {}'.format(results['score']))
        print('Precision: {}'.format(results['precision']))
        print('Recall: {}'.format(results['recall']))
        print('F1: {}'.format(results['f1']))
        print('TP: {}'.format(results['TP']))
        print('FP: {}'.format(results['FP']))
        print('FN: {}'.format(results['FN']))
    else:
        if train_test is None:
            identities = pd.read_csv(common.IDENTITIES, sep=' ', header=None)
            identities.columns = ['image', 'id']

            x = identities['image']
            y = identities['id']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
            train_test = (x_train, x_test, y_train, y_test)

        results, params = optimize(annotations, ground_truths, train_test)

        print('Best parameters:')
        print('     min_neighbors: {}'.format(params['min_neighbors']))
        print('     scale_factor: {}'.format(params['scale_factor']))
        print('     threshold: {}'.format(params['threshold']))
        print('Results:')
        print(results)


if __name__ == '__main__':
    print('Running vj_detector.py...')
    main(eval_once=False, visualize=False, normalize=True)
    print('Done.')
    