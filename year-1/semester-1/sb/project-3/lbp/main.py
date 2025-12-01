# The main module of this project. It contains the implementation of the LBP training 
# and the evaluation of the VJ detector, the LBP classifiers and the pixel2pixel
# recognizer. To run the project, simply run this file as a script, wait for a long
# time and then enjoy the results.

import  cLBP, detector, common, utils
import os, cv2, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances


def compute_pixel2pixel(image: np.ndarray) -> np.ndarray:
    """
    Converts a 2D image into a 1D vector.

    :param image: Image to convert.
    :returns: Converted image.
    """
    return image.reshape(-1)


def optimize_lbp(train_set: tuple) -> tuple:
    """
    Optimizes the library and the custom implementation of the LBP recognizers.

    :param train_set: Train set to optimize the parameters on.
    :returns: Best parameters for the custom and library LBP implementation.
    """
    print('Running optimize_lbp...')

    # Before training, run the VJ detector on the train set to generate the cropped images.
    # Uncomment this if you need to generate the images.
    # detector.main(eval_once=True, visualize=True, normalize=True, train_test=train_set)

    x_train, y_train = train_set
    best_accuracy = 0
    best_parameters = {}

    # Preset certain parameters to speed up the training process. Some of these were 
    # determined by trial and error.
    parameter_configurations = [
        {'P': 4, 'R': 1, 'rdim': (16, 16)},
        {'P': 8, 'R': 1, 'rdim': (16, 16)},
        {'P': 16, 'R': 1, 'rdim': (32, 32)},
        {'P': 8, 'R': 2, 'rdim': (16, 16)},
        {'P': 16, 'R': 2, 'rdim': (32, 32)},
        {'P': 16, 'R': 3, 'rdim': (32, 32)},
        {'P': 24, 'R': 3, 'rdim': (32, 32)},
    ]
    image_dimensions = (600, 500)
    #region_dimensions = (16, 16)

    # Pre-generate the existing images to reduce redundant checks during training.
    print("Generating the image set...")
    existing_train_images = {image for image in os.listdir(common.OUT_IMAGES_CR_COMPUTED) if image.removesuffix('.vj.detected.cropped.png') in set(x_train)}
    image_to_identities = dict(zip(x_train, y_train))
    
    print('Evaluating the custom LBP implementation...')
    for configuration in parameter_configurations:
        # Precompute all LBP feature vectors for the train set with the current parameters.
        P = configuration['P']
        R = configuration['R']
        rdim = configuration['rdim']

        # Compute the LBP feature vectors for the images.
        print("Computing the feature vectors...")
        feature_vectors = dict()
        for count, image_path in enumerate(existing_train_images):
            # Read the image and resize it to the smaller dimensions.
            image = cLBP.load_image(common.OUT_IMAGES_CR_COMPUTED + image_path)
            image = cv2.resize(image, image_dimensions)

            # Compute the LBP feature vector for the image.
            feature_vector, _ = cLBP.compute_lbp(image, rdim, P, R)
            feature_vectors[image_path.removesuffix('.vj.detected.cropped.png')] = feature_vector

            if count % 20 == 0:
                print("     Processed {} images...".format(count))

        # For each image, compute the distance to all other images and find the closest one.
        correct_predictions = 0
        all_predictions = 0
        identities_map = []

        print('Evaluating results...')
        for current_image, current_identity in image_to_identities.items():
            best_distance = np.inf
            best_identity = ''
            for target_image, target_identity in image_to_identities.items():
                if target_image == current_image:
                    continue
                # Skip images for which the VJ detector failed to detect anything.
                if current_image not in feature_vectors.keys() or target_image not in feature_vectors.keys():
                    continue

                # Compute the distance between the feature vectors.
                distance = manhattan_distances(feature_vectors[current_image].reshape(1, -1), feature_vectors[target_image].reshape(1, -1))[0, 0]

                if distance < best_distance:
                    best_distance = distance
                    best_identity = target_identity
            identities_map.append((current_identity, best_identity))
        accuracy = accuracy_score([x[0] for x in identities_map], [x[1] for x in identities_map])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = configuration
    print("Best parameters for the custom LBP implementation: {}".format(best_parameters))

    print('Evaluating the library LBP implementation...')
    for configuration in parameter_configurations:
        # Precompute all LBP feature vectors for the train set with the current parameters.
        P = configuration['P']
        R = configuration['R']

        # Compute the LBP feature vectors for the images.
        print("Computing the feature vectors...")
        feature_vectors = dict()
        for image in existing_train_images:
            # Read the image and resize it to the smaller dimensions.
            image = cLBP.load_image(common.OUT_IMAGES_CR_COMPUTED + image)
            image = cv2.resize(image, image_dimensions)

            # Compute the LBP feature vector for the image.
            feature_vector, _ = cLBP.compute_lbp_lib(image, configuration['rdim'], P, R)
            feature_vectors[image] = feature_vector

        # For each image, compute the distance to all other images and find the closest one.
        correct_predictions = 0
        all_predictions = 0
        identities_map = []

        print('Evaluating results...')
        for current_image, current_identity in image_to_identities.items():
            best_distance = np.inf
            best_identity = ''
            for target_image, target_identity in image_to_identities.items():
                if target_image == current_image:
                    continue
                # Skip images for which the VJ detector failed to detect anything.
                if current_image not in feature_vectors.keys() or target_image not in feature_vectors.keys():
                    continue
                # Compute the distance between the feature vectors.
                distance = manhattan_distances(feature_vectors[current_image].reshape(1, -1), feature_vectors[target_image].reshape(1, -1))[0, 0]

                if distance < best_distance:
                    best_distance = distance
                    best_identity = target_identity
            identities_map.append((current_identity, best_identity))
        accuracy = accuracy_score([x[0] for x in identities_map], [x[1] for x in identities_map])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters_lib = configuration
    print("Best parameters for the library LBP implementation: {}".format(best_parameters))

    return best_parameters, best_parameters_lib


def test_clbp(test_set: tuple, parameters: dict, image_directory: str):
    """
    Tests the custom LBP implementation on the test set.

    :param test_set: Test set to perform the tests on.
    :param parameters: Parameters to use for the LBP implementation.
    :param image_directory: Directory where the images are stored.
    """
    x_test, y_test = test_set
    P = parameters['P']
    R = parameters['R']
    rdim = parameters['rdim']

    existing_test_images = {image for image in os.listdir(image_directory) if image.removesuffix('.vj.detected.cropped.png') in set(x_test)}
    images_to_identities = dict(zip(x_test, y_test))

    accuracy = 0
    correct_predictions = 0
    all_predictions = 0
    identities_map = []
    feature_vectors = dict()

    print("Evaluating the custom LBP implementation...")

    # Compute the LBP feature vectors for the images.
    print("Computing the feature vectors...")
    for count, image_path in enumerate(existing_test_images):
        # Read the image and resize it to the smaller dimensions.
        image = cLBP.load_image(image_directory + image_path)
        image = cv2.resize(image, (600, 500))

        # Compute the LBP feature vector for the image.
        feature_vector, _ = cLBP.compute_lbp(image, rdim, P, R)
        feature_vectors[image_path.removesuffix('.vj.detected.cropped.png')] = feature_vector

        if count % 20 == 0:
            print("     Processed {} images...".format(count))

    # For each image, compute the distance to all other images and find the closest one.
    print('Evaluating results...')
    for current_image, current_identity in images_to_identities.items():
        best_distance = np.inf
        best_identity = ''
        if current_image not in feature_vectors.keys():
            continue
        for target_image, target_identity in images_to_identities.items():
            if target_image == current_image:
                continue
            # Skip images for which the VJ detector failed to detect anything.
            if current_image not in feature_vectors.keys() or target_image not in feature_vectors.keys():
                continue
            # Compute the distance between the feature vectors.
            distance = manhattan_distances(feature_vectors[current_image].reshape(1, -1), feature_vectors[target_image].reshape(1, -1))[0, 0]

            if distance < best_distance:
                best_distance = distance
                best_identity = target_identity
        identities_map.append((current_identity, best_identity))
    true_labels = [str(x[0]).strip() for x in identities_map]
    pred_labels = [str(x[1]).strip() for x in identities_map]
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print("Evaluated the custom LBP implementation on the test:")
    print("     Accuracy: {}".format(accuracy))
    print("     Precision: {}".format(precision))
    print("     Recall: {}".format(recall))
    print("     F1: {}".format(f1))


def test_lbp_lib(test_set: tuple, parameters: dict, image_directory: str):
    """
    Tests the library LBP implementation on the test set.

    :param test_set: Test set to perform the tests on.
    :param parameters: Parameters to use for the LBP implementation.
    :param image_directory: Directory where the images are stored.
    """
    x_test, y_test = test_set
    P = parameters['P']
    R = parameters['R']
    rdim = parameters['rdim']

    existing_test_images = {image for image in os.listdir(image_directory) if image.removesuffix('.vj.detected.cropped.png') in set(x_test)}
    images_to_identities = dict(zip(x_test, y_test))

    accuracy = 0
    correct_predictions = 0
    all_predictions = 0
    identities_map = []
    feature_vectors = dict()

    print("Evaluating the library LBP implementation...")

    # Compute the LBP feature vectors for the images.
    print("Computing the feature vectors...")
    for count, image_path in enumerate(existing_test_images):
        # Read the image and resize it to the smaller dimensions.
        image = cLBP.load_image(image_directory + image_path)
        image = cv2.resize(image, (600, 500))

        # Compute the LBP feature vector for the image.
        feature_vector, _ = cLBP.compute_lbp_lib(image, rdim, P, R)
        feature_vectors[image_path.removesuffix('.vj.detected.cropped.png')] = feature_vector

        if count % 20 == 0:
            print("     Processed {} images...".format(count))

    # For each image, compute the distance to all other images and find the closest one.
    print('Evaluating results...')
    for current_image, current_identity in images_to_identities.items():
        best_distance = np.inf
        best_identity = ''
        if current_image not in feature_vectors.keys():
            continue
        for target_image, target_identity in images_to_identities.items():
            if target_image == current_image:
                continue
            # Skip images for which the VJ detector failed to detect anything.
            if current_image not in feature_vectors.keys() or target_image not in feature_vectors.keys():
                continue
            # Compute the distance between the feature vectors.
            distance = manhattan_distances(feature_vectors[current_image].reshape(1, -1), feature_vectors[target_image].reshape(1, -1))[0, 0]

            if distance < best_distance:
                best_distance = distance
                best_identity = target_identity
        identities_map.append((current_identity, best_identity))
    true_labels = [str(x[0]).strip() for x in identities_map]
    pred_labels = [str(x[1]).strip() for x in identities_map]
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print("Evaluated the library LBP implementation on the test:")
    print("     Accuracy: {}".format(accuracy))
    print("     Precision: {}".format(precision))
    print("     Recall: {}".format(recall))
    print("     F1: {}".format(f1))


def test_pixel2pixel(test_set: tuple, image_directory: str):
    """
    Tests the pixel2pixel recognizer on the test set.

    :param test_set: Test set to perform the tests on.
    :param image_directory: Directory where the images are stored.
    """
    x_test, y_test = test_set

    existing_test_images = {image for image in os.listdir(image_directory) if image.removesuffix('.vj.detected.cropped.png') in set(x_test)}
    images_to_identities = dict(zip(x_test, y_test))

    accuracy = 0
    correct_predictions = 0
    all_predictions = 0
    identities_map = []
    feature_vectors = dict()

    print("Evaluating the pixel2pixel recognizer...")

    # Compute the LBP feature vectors for the images.
    print("Computing the feature vectors...")
    for count, image_path in enumerate(existing_test_images):
        # Read the image and resize it to the smaller dimensions.
        image = cLBP.load_image(image_directory + image_path)
        image = cv2.resize(image, (600, 500))

        # Compute the LBP feature vector for the image.
        feature_vector = compute_pixel2pixel(image)
        feature_vectors[image_path.removesuffix('.vj.detected.cropped.png')] = feature_vector

        if count % 20 == 0:
            print("     Processed {} images...".format(count))

    # For each image, compute the distance to all other images and find the closest one.
    print('Evaluating results...')
    for current_image, current_identity in images_to_identities.items():
        best_distance = np.inf
        best_identity = ''
        if current_image not in feature_vectors.keys():
            continue
        for target_image, target_identity in images_to_identities.items():
            if target_image == current_image:
                continue
            # Skip images for which the VJ detector failed to detect anything.
            if current_image not in feature_vectors.keys() or target_image not in feature_vectors.keys():
                continue
            # Compute the distance between the feature vectors.
            distance = manhattan_distances(feature_vectors[current_image].reshape(1, -1), feature_vectors[target_image].reshape(1, -1))[0, 0]

            if distance < best_distance:
                best_distance = distance
                best_identity = target_identity
        identities_map.append((current_identity, best_identity))
    true_labels = [str(x[0]).strip() for x in identities_map]
    pred_labels = [str(x[1]).strip() for x in identities_map]
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print("Evaluated the pixel2pixel recognizer on the test:")
    print("     Accuracy: {}".format(accuracy))
    print("     Precision: {}".format(precision))
    print("     Recall: {}".format(recall))
    print("     F1: {}".format(f1))


def tests_computed(test_set: tuple, best_parameters: dict, best_parameters_lib: dict):
    """
    Performs the LBP tests on the images computed by the VJ detector.
    
    :param test_set: Test set to perform the tests on.
    :param best_parameters: Best parameters for the custom LBP implementation.
    :param best_parameters_lib: Best parameters for the library LBP implementation.
    """
    print('Running tests_computed...')

    # Perform the test on the custom LBP implementation.
    test_clbp(test_set, best_parameters, common.OUT_IMAGES_CR_COMPUTED)

    # Perform the test on the library LBP implementation.
    test_lbp_lib(test_set, best_parameters_lib, common.OUT_IMAGES_CR_COMPUTED)

    # Perform the test on the pixel2pixel recognizer.
    test_pixel2pixel(test_set, common.OUT_IMAGES_CR_COMPUTED)


def tests_ground_truths(test_set: tuple, best_parameters: dict, best_parameters_lib: dict):
    """
    Performs the LBP tests on the ground truths.

    :param test_set: Test set to perform the tests on.
    :param best_parameters: Best parameters for the custom LBP implementation.
    :param best_parameters_lib: Best parameters for the library LBP implementation.
    """    
    print('Running tests_ground_truths...')

    # Perform the test on the custom LBP implementation.
    test_clbp(test_set, best_parameters, common.OUT_IMAGES_CR_TRUTHS)

    # Perform the test on the library LBP implementation.
    test_lbp_lib(test_set, best_parameters_lib, common.OUT_IMAGES_CR_TRUTHS)

    # Perform the test on the pixel2pixel recognizer.
    test_pixel2pixel(test_set, common.OUT_IMAGES_CR_TRUTHS)


def tests(test_set: tuple, best_parameters: dict, best_parameters_lib: dict):
    """
    Performs the tests for the VJ detector, custom LBP implementation, library LBP 
    implementation and the pixel2pixel recognizer on the optimized parameters for 
    all of them.

    :param test_set: Test set to perform the tests on.
    :param best_parameters: Best parameters for the custom LBP implementation.
    :param best_parameters_lib: Best parameters for the library LBP implementation.
    """
    print('Running tests...')

    # Perform the test on the VJ detector.
    #detector.main(eval_once=True, visualize=True, normalize=True, train_test=test_set)

    tests_computed(test_set, best_parameters, best_parameters_lib)
    tests_ground_truths(test_set, best_parameters, best_parameters_lib)


if __name__ == '__main__':
    print('Running main...')

    print('Generating train-test split...')
    # Generate the train-test split on which to evaluate the detector and classifier.
    identities = pd.read_csv(common.IDENTITIES, sep=' ', header=None)
    identities.columns = ['image', 'id']

    x = identities['image']
    y = identities['id']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    
    train_set = (x_train, y_train)
    test_set = (x_test, y_test)

    # Optimize the parameters for the custom and library LBP implementation.
    #best_parameters, best_parameters_lib = optimize_lbp(train_set)

    best_parameters = {'P': 8, 'R': 1, 'rdim': (16, 16)}
    best_parameters_lib = {'P': 8, 'R': 1, 'rdim': (16, 16)}

    # Perform all the tests and print results.
    tests(test_set, best_parameters, best_parameters_lib)
