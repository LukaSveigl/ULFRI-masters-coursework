import numpy as np
import cv2

from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak
from tracker import Tracker

class BaseMOSSETracker(Tracker):
    """
    A basic MOSSE tracker implementation.

    Args:
        Tracker (class): The base tracker class.
    """

    def __init__(self):
        """
        Initializes the MOSSE tracker with the given parameters.
        """
        self.parameters = MOSSEParams()
        super().__init__()

    def name(self) -> str:
        """
        Returns the name of this tracker, which is used by the evaluation toolkit.

        Returns:
            str: The name of the tracker.
        """
        return "base_mosse2"
    
    def initialize(self, image: np.ndarray, region: np.ndarray):
        """
        Initializes the MOSSE tracker with the given image and region. Besides the two parameters,
        the tracker also initializes the cosine window and the Gaussian peak.

        Args:
            image (np.ndarray): The initial image.
            region (np.ndarray): The initial region.
        """
        # Convert the region from 8 to 4 coordinates (x, y, width, height).
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the starting position using the equation (x + width / 2, y + height / 2).
        self.start_position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.region_size = (int(region[2]), int(region[3]))

        # Fix the region sizes to the nearest lowest odd number.
        if self.region_size[0] % 2 == 0:
            self.region_size = (self.region_size[0] - 1, self.region_size[1])
        if self.region_size[1] % 2 == 0:
            self.region_size = (self.region_size[0], self.region_size[1] - 1)

        # Enlarge the region size and fix it to the nearest lowest odd number.
        self.enlarged_region_size = (
            int(self.region_size[0] * self.parameters.enlarge_factor), int(self.region_size[1] * self.parameters.enlarge_factor)
        )
        if self.enlarged_region_size[0] % 2 == 0:
            self.enlarged_region_size = (self.enlarged_region_size[0] - 1, self.enlarged_region_size[1])
        if self.enlarged_region_size[1] % 2 == 0:
            self.enlarged_region_size = (self.enlarged_region_size[0], self.enlarged_region_size[1] - 1)

        # Create the cosine window and the Gaussian peak.
        self.cosine_window = create_cosine_window(self.enlarged_region_size)
        self.gauss_peak = create_gauss_peak(self.enlarged_region_size, self.parameters.sigma)

        patch, mask = get_patch(gray_image, self.start_position, self.enlarged_region_size)

        # Compute the Fourier transforms of the patch and the Gaussian peak, along with their conjugates.
        patch = patch * self.cosine_window
        self.F_hat = np.fft.fft2(patch)
        self.F_hat_conj = np.conj(self.F_hat)
        self.G_hat = np.fft.fft2(self.gauss_peak)

        # Compute the H_hat matrix using the equation: H_hat = F_hat_conj * G_hat / (F_hat_conj * F_hat + lambda).
        #self.H_hat = np.multiply(self.F_hat_conj, self.G_hat) / (np.multiply(self.F_hat_conj, self.F_hat) + self.parameters.lambd)
        self.H_hat = np.multiply(self.G_hat, self.F_hat_conj) / (np.multiply(self.F_hat, self.F_hat_conj) + self.parameters.lambd)

    def track(self, image: np.ndarray) -> np.ndarray:
        """
        Tracks the object in the given image.

        Args:
            image (np.ndarray): The image to track.

        Returns:
            np.ndarray: The tracked region.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patch, mask = get_patch(gray_image, self.start_position, self.enlarged_region_size)
        patch = patch * self.cosine_window

        # Compute the cross-power spectrum using the equation: G = H_hat * F_hat.
        self.F_hat = np.fft.fft2(patch)
        
        H_hat_F_hat = np.multiply(self.H_hat, self.F_hat)
        R = np.fft.ifft2(H_hat_F_hat)

        # Update the starting position using the peak of the cross-correlation.
        y, x = np.unravel_index(np.argmax(np.abs(R), axis=None), R.shape)

        if x > self.enlarged_region_size[0] / 2: 
            x -= self.enlarged_region_size[0]
        if y > self.enlarged_region_size[1] / 2:
            y -= self.enlarged_region_size[1]

        self.start_position = (self.start_position[0] + x, self.start_position[1] + y)

        # Using the new starting position, update the tracker matrices.
        patch, mask = get_patch(gray_image, self.start_position, self.enlarged_region_size)
        self._update_tracker_matrices(patch)

        # Return the tracked region.
        tracked_x = self.start_position[0] - self.region_size[0] / 2
        tracked_y = self.start_position[1] - self.region_size[1] / 2
        return [tracked_x, tracked_y, self.region_size[0], self.region_size[1]]

    def _update_tracker_matrices(self, patch: np.ndarray):
        """
        Updates the tracker matrices using the given patch.

        Args:
            patch (np.ndarray): The patch to update the matrices with.
        """
        patch = patch * self.cosine_window

        self.F_hat = np.fft.fft2(patch)
        self.F_hat_conj = np.conj(self.F_hat)

        # Compute the H_hat matrix using the equation: H_hat = F_hat_conj * G_hat / (F_hat_conj * F_hat + lambda).
        #H_hat_new = np.multiply(self.F_hat_conj, self.G_hat) / (np.multiply(self.F_hat_conj, self.F_hat) + self.parameters.lambd)
        H_hat_new = np.multiply(self.G_hat, self.F_hat_conj) / (np.multiply(self.F_hat, self.F_hat_conj) + self.parameters.lambd)

        # Adjust the H_hat matrix using the learning rate alpha.
        self.H_hat = (1 - self.parameters.alpha) * self.H_hat + self.parameters.alpha * H_hat_new        


class MOSSEParams():
    """
    Parameters for the MOSSE tracker.
    """

    def __init__(self):
        """
        Initializes the MOSEE parameters.
        """
        # self.alpha = 0.125
        # self.sigma = 2.0
        # self.lambd = 1e-3
        # self.enlarge_factor = 1.5

        # self.alpha = 0.15
        # self.sigma = 2.0
        # self.lambd = 1e-3
        # self.enlarge_factor = 1.2

        # self.alpha = 0.2
        # self.sigma = 2.0
        # self.lambd = 1e-4
        # self.enlarge_factor = 1.2

        # self.alpha = 0.22
        # self.sigma = 2.05
        # self.lambd = 1e-4
        # self.enlarge_factor = 1.2

        # Best combination, 71 fails
        # self.alpha = 0.215
        # self.sigma = 2.05
        # self.lambd = 1e-4
        # self.enlarge_factor = 1.2

        # self.alpha = 0.23
        # self.sigma = 2.05
        # self.lambd = 1e-4
        # self.enlarge_factor = 1.2

        # self.alpha = 0.215
        # self.sigma = 2.05
        # self.lambd = 1e-4
        # self.enlarge_factor = 1.2

        # self.alpha = 0.1
        # self.sigma = 1.0
        # self.lambd = 1e-5
        # self.enlarge_factor = 1.0

        self.alpha = 0.23
        self.sigma = 1.4
        self.lambd = 1e-2
        self.enlarge_factor = 1.15
