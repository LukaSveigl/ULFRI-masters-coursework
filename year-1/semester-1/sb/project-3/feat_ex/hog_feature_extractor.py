import numpy as np
import cv2
import os


def _compute_gradients(image: np.ndarray) -> tuple:
    """
    Computes the gradients of the given image.

    :param image: The image to compute the gradients for.
    :return: The gradients of the given image.
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    direction = np.arctan2(gradient_y, gradient_x)

    return magnitude, direction


def _compute_hog_cell_histogram(magnitude, direction, orientations):
    """
    Computes the HOG histogram for the given cell.

    :param magnitude:    The magnitude of the gradients.
    :param direction:    The direction of the gradients.
    :param orientations: The number of orientations to consider.
    :return: The HOG histogram for the given cell.
    """
    bins = np.linspace(0, np.pi, orientations + 1)
    histogram, _ = np.histogram(direction, bins=bins, weights=magnitude)
    return histogram


def hog(image_path: str, **kwargs: dict) -> np.ndarray:
    """
    Computes the HOG features for the given image.

    :param image_path: The path to the image to compute the HOG features for.
    :param kwargs:     The keyword arguments.
    :return: The HOG features for the given image.
    """

    orientations = kwargs.get('orientations', 9)
    pixels_per_cell = kwargs.get('pixels_per_cell', (8, 8))
    cells_per_block = kwargs.get('cells_per_block', (3, 3))

    # Convert image to grayscale if necessary.
    if len(image_path.shape) == 3:
        image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

    # Compute gradients.
    magnitude, direction = _compute_gradients(image)

    # Define the number of cells in the image.
    cells_x = image.shape[1] // pixels_per_cell[1]
    cells_y = image.shape[0] // pixels_per_cell[0]

    # Compute the histograms for each cell.
    hog_features = []

    for y in range(cells_y):
        for x in range(cells_x):
            cell_magnitude = magnitude[y * pixels_per_cell[0]: (y + 1) * pixels_per_cell[0],
                                       x * pixels_per_cell[1]: (x + 1) * pixels_per_cell[1]]
            cell_direction = direction[y * pixels_per_cell[0]: (y + 1) * pixels_per_cell[0],
                                       x * pixels_per_cell[1]: (x + 1) * pixels_per_cell[1]]
            hog_features.append(_compute_hog_cell_histogram(cell_magnitude, cell_direction, orientations))

    # TODO: Add block normalization.

    return np.array(hog_features).flatten()


def hog_tune_params(image_path: str, optimizer, **kwargs: dict) -> np.ndarray:
    """
    Tunes the parameters of the HOG feature extractor with the given optimizer.

    :param image_path: The path to the image to use for tuning.
    :param optimizer:  The optimizer to use for tuning.
    :param kwargs:     The keyword arguments to pass to the HOG feature extractor.
    :return: The tuned parameters.
    """
    pass


def hog_runner(mode: str):
    """
    Runs the HOG feature extractor.

    :param mode: The mode to run the HOG feature extractor in.
    """
    if mode not in {'train', 'test', 'val'}:
        raise ValueError('Invalid mode')

    folder_dir = '../datasets/ears/images/' + mode
    features_list = []

    for filename in os.listdir(folder_dir):
        print("Processing image: " + filename)
        image = cv2.imread(os.path.join(folder_dir, filename))
        image = cv2.resize(image, (128, 128))
        features = hog(image)

        features_list.append(features)

        filename = filename.replace('images', 'features-hog')
        filename = filename.replace('.png', '.txt')

        features_folder_dir = folder_dir.replace('images', 'features-hog')

        np.savetxt(os.path.join(features_folder_dir, filename), features, delimiter=',')

        del features

    print(features_list)


if __name__ == "__main__":
    hog_runner('train')

