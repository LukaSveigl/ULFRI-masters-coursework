# This file contains the implementation of a simple LBP detector, described in the 
# paper: http://vision.stanford.edu/teaching/cs231b_spring1415/papers/lbp.pdf. The 
# algorithm is not implemented completely, but atleast rotation invariance is ensured.
# To use this detector, either run this file as a script, which will run a simple
# evaluation, or import it elsewhere and run the `compute_lbp` function.

import utils, common
import cv2, skimage as sk, numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

def load_image(src_path: str) -> np.ndarray:
    """
    Loads the image from the given path and preprocesses it.

    :param src_path: Path to the image.
    :returns: Preprocessed image.
    """
    image = cv2.imread(src_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def is_uniform(neighbors: list) -> bool:
    """
    Checks if the given list of neighbors is uniform.

    :param neighbors: List of neighbors.
    :returns: True if the list is uniform, False otherwise.
    """
    transitions = 0
    for i in range(len(neighbors)):
        if neighbors[i] != neighbors[(i + 1) % len(neighbors)]:
            transitions += 1
    return transitions <= 2


def compute_lbp_neighborhood(image: np.ndarray, point: tuple, P: int, R) -> list:
    """
    Computes the LBP value for the given neighborhood, while ensuring rotation invariance 
    by checking if the neighborhood is uniform.

    :param image: Image to compute the LBP value for.
    :param point: Point to compute the LBP value for.
    :param P: Number of neighbors to consider.
    :param R: Radius of circle to consider.
    :returns: LBP value.
    """
    lbp_value = 0
    point_x, point_y = point
    center_value = image[point_y, point_x]
    neighborhood = []
    for i in range(P):
        x = int(point_x + R * np.cos(2 * np.pi * i / P))
        y = int(point_y - R * np.sin(2 * np.pi * i / P))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if image[y, x] >= center_value:
                lbp_value |= (1 << i)
                neighborhood.append(1)
            else:
                neighborhood.append(0)
    if is_uniform(neighborhood):
        return lbp_value
    else:
        return 2 ** P
    

def compute_lbp_by_region(image: np.ndarray, P: int, R: int) -> np.ndarray:
    """
    Computes the LBP values for the given region.

    :param image: The region of the image to compute the LBP values for.
    :param P: Number of neighbors to consider.
    :param R: Radius of circle to consider.
    :returns: LBP computed region.
    """
    lbp_image = np.zeros_like(image, dtype=np.uint32)
    padded_image = np.pad(image, R, mode='constant', constant_values=0)
    for y in range(R, padded_image.shape[0] - R):
        for x in range(R, padded_image.shape[1] - R):
            lbp_value = compute_lbp_neighborhood(padded_image, (x, y), P, R)
            lbp_image[y - R - 1, x - R - 1] = lbp_value
    return lbp_image


def compute_lbp_histogram(region: np.ndarray, P: int) -> np.ndarray:
    """
    Computes the LBP histogram for the given region.

    :param region: The region of the image to compute the LBP histogram for.
    :param P: Number of neighbors to consider.
    :returns: LBP histogram.
    """
    histogram = np.zeros(2 ** P, dtype=np.int32)
    for y in range(region.shape[0]):
        for x in range(region.shape[1]):
            if region[y, x] != 2 ** P:
                histogram[region[y, x]] += 1
    return histogram


def compute_lbp(image: np.ndarray, region_dimensions: tuple, P: int, R: int) -> tuple:
    """
    Computes the LBP image and it's feature vector for the given image.

    :param image: Image to compute the LBP image and feature vector for.
    :param region_dimensions: Dimensions of the region by which to split the image.
    :param P: Number of neighbors to consider.
    :param R: Radius of circle to consider.
    :returns: Tuple of the feature vector and the LBP image.
    """
    feature_vector = np.array([], dtype=np.uint32)
    r_height, r_width = region_dimensions
    full_lbp_image = np.zeros_like(image, dtype=np.uint32)
    num_regions = 0
    for y in range(0, image.shape[0], r_height):
        for x in range(0, image.shape[1], r_width):
            num_regions += 1
            # Extract the region and compute the LBP values, store
            # them in the full LBP image and compute the histogram.
            region = image[y:y + r_height, x:x + r_width]
            lbp_image = compute_lbp_by_region(region, P, R)
            full_lbp_image[y:y + r_height, x:x + r_width] = lbp_image
            histogram = compute_lbp_histogram(lbp_image, P)
            feature_vector = np.append(feature_vector, histogram)
    return feature_vector, full_lbp_image


def compute_lbp_lib(image: np.ndarray, region_dimensions: tuple, P: int, R: int) -> tuple:
    """
    Same as `compute_lbp`, but instead of using the custom implementation, it uses the one from scikit-image.

    :param image: Image to compute the LBP image and feature vector for.
    :param region_dimensions: Dimensions of the region by which to split the image.
    :param P: Number of neighbors to consider.
    :param R: Radius of circle to consider.
    :returns: Tuple of the feature vector and the LBP image.
    """
    feature_vector = np.array([], dtype=np.uint32)
    r_height, r_width = region_dimensions
    full_lbp_image = np.zeros_like(image, dtype=np.uint32)
    for y in range(0, image.shape[0], r_height):
        for x in range(0, image.shape[1], r_width):
            region = image[y:y + r_height, x:x + r_width]
            lbp_image = sk.feature.local_binary_pattern(region, P, R, method='default')
            # The result of skimage's LBP is a float image, so we need to convert it to uint32.
            lbp_image = lbp_image.astype(np.uint32)
            full_lbp_image[y:y + r_height, x:x + r_width] = lbp_image
            histogram = compute_lbp_histogram(lbp_image.astype(np.uint32), P)
            feature_vector = np.append(feature_vector, histogram)
    return np.array(feature_vector), full_lbp_image


if __name__ == '__main__':
    image = load_image(common.OUT_IMAGES_CR_TRUTHS + '0501.png')

    # Create an output image for LBP representation
    lbp_image = np.zeros_like(image)
    P = 8  # Number of neighbors to consider
    R = 1  # Radius of circle to consider

    lbp_image = compute_lbp_by_region(image, P, R)

    lbp_fv, lbp_image = compute_lbp(image, (16, 16), P, R)

    hist = np.zeros(256, dtype=np.int32)
    image2 = cv2.imread(common.OUT_IMAGES_CR_TRUTHS + '0501.png', cv2.IMREAD_GRAYSCALE)
    image3 = image2 / 255.0
    lbp_fvl, lbp_skimage = compute_lbp_lib(image3, (16, 16), P, R)

    print(euclidean_distances(lbp_fv.reshape(1, -1), lbp_fvl.reshape(1, -1)))
    print(cosine_similarity(lbp_fv.reshape(1, -1), lbp_fvl.reshape(1, -1)))
    print(manhattan_distances(lbp_fv.reshape(1, -1), lbp_fvl.reshape(1, -1)))

    # Display the LBP image
    cv2.imshow('LBP Image', lbp_skimage)
    cv2.imshow('LBP Image2', lbp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()