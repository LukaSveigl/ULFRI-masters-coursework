import numpy as np
import cv2

from ex2_utils import Tracker, get_patch, extract_histogram, create_epanechnik_kernel, backproject_histogram

class MeanShiftTracker(Tracker):
    """
    A Mean-Shift tracker.

    Args:
        Tracker (class): The base tracker class.
    """

    def initialize(self, image: np.ndarray, region: np.ndarray):
        """
        Initializes the Mean-Shift tracker with the given image and region. The tracker is initialized with the
        starting position, region size, histogram, Epanechnikov kernel and meshgrid.

        Args:
            image (np.ndarray): The initial image.
            region (np.ndarray): The initial region.
        """
        # Convert the region from 8 to 4 coordinates (x, y, width, height).
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # Compute the starting position using the equation (x + width / 2, y + height / 2).
        self.start_position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.region_size = (int(region[2]), int(region[3]))

        # Fix the region sizes to the nearest lowest odd number.
        if self.region_size[0] % 2 == 0:
            self.region_size = (self.region_size[0] - 1, self.region_size[1])
        if self.region_size[1] % 2 == 0:
            self.region_size = (self.region_size[0], self.region_size[1] - 1)

        # Create the Epanechnikov kernel.
        self.epanechnik_kernel = create_epanechnik_kernel(self.region_size[0], self.region_size[1], self.parameters.sigma)
        half_width = self.region_size[0] // 2
        half_height = self.region_size[1] // 2
        self.X, self.Y = np.meshgrid(np.arange(-half_width, half_width + 1), np.arange(-half_height, half_height + 1))

        # Extract the histogram of the region and normalize it.
        patch, mask = get_patch(image, self.start_position, self.region_size)
        self.histogram = self._extract_and_normalize_histogram(patch)

    def track(self, image: np.ndarray) -> np.ndarray:
        """
        Tracks the object in the given image.

        Args:
            image (np.ndarray): The image to track.

        Returns:
            np.ndarray: The tracked region.
        """
        for _ in range(self.parameters.max_iterations):
            patch, mask = get_patch(image, self.start_position, self.region_size)
            patch = patch.astype(np.float32)

            next_frame_histogram = self._extract_and_normalize_histogram(patch)

            # Compute the weights using the equation v = sqrt(q / p + epsilon) and backproject the histogram.
            v = np.sqrt(self.histogram / (next_frame_histogram + self.parameters.epsilon))
            backprojected_histogram = backproject_histogram(patch, v, self.parameters.histogram_bins)
            backprojected_histogram /= np.sum(backprojected_histogram)

            x_change = np.sum(backprojected_histogram * self.X)
            y_change = np.sum(backprojected_histogram * self.Y)

            # Update the starting position using the equation (x + x_change, y + y_change).
            new_position = (self.start_position[0] + x_change, self.start_position[1] + y_change)

            if np.linalg.norm(np.array(new_position) - np.array(self.start_position)) < self.parameters.threshold:
                break

            self.start_position = new_position

        # Update the internal histogram for use in the next call of this method using the equation:
        # histogram = (1 - alpha) * q + alpha * q_tilda.
        self.histogram = (1 - self.parameters.alpha) * self.histogram + self.parameters.alpha * next_frame_histogram

        tracked_x = self.start_position[0] - self.region_size[0] / 2
        tracked_y = self.start_position[1] - self.region_size[1] / 2
        return [tracked_x, tracked_y, self.region_size[0], self.region_size[1]]

    def _extract_and_normalize_histogram(self, patch: np.ndarray) -> np.ndarray:
        """
        Extracts and normalizes the histogram of the given patch.

        Args:
            patch (np.ndarray): The patch to extract the histogram from.

        Returns:
            np.ndarray: The normalized histogram.
        """
        histogram = extract_histogram(patch, self.parameters.histogram_bins, self.epanechnik_kernel)
        histogram /= np.sum(histogram)
        return histogram


class MSParams():
    """
    Parameters for the Mean-Shift tracker.
    """

    def __init__(self):
        """
        Initializes the Mean-Shift parameters.
        """
        self.histogram_bins = 16
        self.alpha = 0.01
        self.sigma = 1
        self.epsilon = 1e-3
        self.threshold = 0.5
        self.max_iterations = 40
