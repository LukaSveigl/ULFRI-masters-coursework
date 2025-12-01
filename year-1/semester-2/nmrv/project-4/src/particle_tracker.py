import numpy as np
import cv2
import time

from ex2_utils import get_patch, extract_histogram, create_epanechnik_kernel# , Tracker
from ex4_utils import kalman_step, gaussian_prob, sample_gauss
from tracker import Tracker

from motion_models import random_walk, nearly_constant_velocity, nearly_constant_acceleration

class ParticleTracker(Tracker):
    """
    A basic particle tracker implementation.

    Args:
        Tracker (class): The base tracker class.
    """

    def __init__(self):
        """
        Initializes the particle tracker with the given parameters.
        """
        # Check if the Tracker's __init__ method accepts a parameter called params.
        if "params" in Tracker.__init__.__code__.co_varnames:
            super().__init__(params=ParticleParams())
        else:
            self.parameters = ParticleParams()
        #super().__init__(params=ParticleParams())

    def name(self) -> str:
        """
        Returns the name of this tracker, which is used by the evaluation toolkit.

        Returns:
            str: The name of the tracker.
        """
        return "particle_tracker"
    
    def initialize(self, image: np.ndarray, region: np.ndarray):
        """
        Initializes the particle tracker with the given image and region. Besides the two parameters,
        the tracker also initializes the system matrix, system covariance, and the initial histogram.

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

        #self.parameters.q = np.mean(self.region_size) * self.parameters.q_percentage
        #self.parameters.q = np.max(self.region_size) * self.parameters.q_percentage
        self.parameters.q = np.min(self.region_size) * self.parameters.q_percentage
        # Extract the system matrix and system covariance from the motion model.
        self.Fi, _, self.Q, _ = self.parameters.motion_model(self.parameters.q, self.parameters.r)

        self.epanechnik_kernel = create_epanechnik_kernel(self.region_size[0], self.region_size[1], 1)# self.parameters.sigma)
        patch, mask = get_patch(image, self.start_position, self.region_size)
        self.histogram = self._extract_and_normalize_histogram(patch)

        # Generate the particle state.
        self.particles = np.zeros(self.Fi.shape[0])
        self.particles[0] = self.start_position[0]
        self.particles[1] = self.start_position[1]
        self.X = sample_gauss(np.zeros(len(self.Fi)), self.Q, self.parameters.N) + self.particles[np.newaxis,:]
        self.W = np.ones(self.parameters.N) / self.parameters.N

    def track(self, image: np.ndarray) -> np.ndarray:
        """
        Tracks the object in the given image.

        Args:
            image (np.ndarray): The image to track.

        Returns:
            np.ndarray: The tracked region.
        """
        height, width = image.shape[:2]
        
        # Update the particle state using the motion model.
        self._update_particle_state()

        histogram_created = False

        for i, x in enumerate(self.X):
            x, y = x[0], x[1]
            # If the particle is outside the image, skip it.
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            patch, mask = get_patch(image, (x, y), self.region_size)
            histogram_created = True
            histogram = self._extract_and_normalize_histogram(patch)
            self._update_particle_weight(i, histogram)
        
        self.W /= np.sum(self.W)
        self.start_position = self.W @ self.X[:, :2]

        if histogram_created:
            self.histogram = self.parameters.alpha * histogram + (1 - self.parameters.alpha) * self.histogram

        # Return the tracked region.
        tracked_x = self.start_position[0] - self.region_size[0] / 2
        tracked_y = self.start_position[1] - self.region_size[1] / 2
        return [tracked_x, tracked_y, self.region_size[0], self.region_size[1]]

    def _update_particle_state(self):
        """
        Updates the particle state using the motion model using the code provided in the assignment.
        """
        cumsum_weights = np.cumsum(self.W)
        rand_samples = np.random.rand(self.parameters.N, 1)
        sampled_indices = np.digitize(rand_samples, cumsum_weights)

        # Resample the particles.
        self.X = self.X[sampled_indices.flatten(), :]
        
        # Update the particle state using the motion model.
        self.X = self.X @ self.Fi.T

        # Add noise to the particles.
        self.X += sample_gauss(np.zeros(len(self.Fi)), self.Q, self.parameters.N)

    def _update_particle_weight(self, i: int, histogram: np.ndarray):
        """
        Updates the weight of the particle at the given index.

        Args:
            i (int): The index of the particle.
            histogram (np.ndarray): The histogram of the given particle.
        """
        def hellinger(p: np.ndarray, q: np.ndarray) -> float:
            """
            Computes the Hellinger distance between two histograms.
            https://en.wikipedia.org/wiki/Hellinger_distance

            Args:
                p (np.ndarray): The first histogram.
                q (np.ndarray): The second histogram.

            Returns:
                float: The Hellinger distance between the two histograms.
            """
            return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    
        # Update the weight of the particle using the equation: w_i = p_i = e ^ (-1/2 * ((d_i^(hel)^2) / sigma^2))
        self.W[i] = np.exp(((-1/2) * (hellinger(self.histogram, histogram) ** 2)) / (self.parameters.sigma ** 2))


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


class ParticleParams():
    """
    Parameters for the MOSSE tracker.
    """

    def __init__(self):
        """
        Initializes the MOSEE parameters.
        """
        #self.alpha = 0.01
        #self.alpha = 0.1
        self.alpha = 0.01 #0.05 ! best
        self.sigma = 0.26 #0.25 ! best
        self.histogram_bins = 16
        #self.N = 70
        self.N = 100
        #self.q = 0.3
        #self.q_percentage = 0.01
        #self.q_percentage = 0.01
        #self.q_percentage = 0.03 ! best
        self.q_percentage = 0.03
        self.r = 1
        #self.motion_model = nearly_constant_velocity
        self.motion_model = nearly_constant_acceleration
        self.seed = np.random.seed(1) #np.random.seed(1)

        # Current best -> 67 fails:
        # alpha = 0.05
        # sigma = 0.20
        # histogram_bins = 16
        # N = 100
        # q_percentage = 0.03

        # Current best -> 61 fails:
        # alpha = 0.01
        # sigma = 0.20
        # histogram_bins = 16
        # N = 100
        # q_percentage = 0.03

        # Current best -> 58 fails:
        # alpha = 0.01
        # sigma = 0.22
        # histogram_bins = 16
        # N = 100
        # q_percentage = 0.03

        # Current best -> 54 fails:
        # alpha = 0.01
        # sigma = 0.25
        # histogram_bins = 16
        # N = 100
        # q_percentage = 0.03

        # Current best -> 51 fails:
        # alpha = 0.01
        # sigma = 0.26
        # histogram_bins = 16
        # N = 100
        # q_percentage = 0.03


        # --------------------

        # Current best -> 62 fails:
        # alpha = 0.01
        # sigma = 0.22
        # histogram_bins = 16
        # N = 120
        # q_percentage = 0.03

        # Current best -> 59 fails:
        # alpha = 0.01
        # sigma = 0.225
        # histogram_bins = 16
        # N = 120
        # q_percentage = 0.03

        # Current best -> 59 fails:
        # alpha = 0.001
        # sigma = 0.225
        # histogram_bins = 16
        # N = 120
        # q_percentage = 0.03

        # --------------------

        # Current best -> 65 fails:
        # alpha = 0.1
        # sigma = 0.22
        # histogram_bins = 16
        # N = 115
        # q_percentage = 0.03

        # Current best -> 63 fails:
        # alpha = 0.1
        # sigma = 0.22
        # histogram_bins = 16
        # N = 115
        # q_percentage = 0.03
