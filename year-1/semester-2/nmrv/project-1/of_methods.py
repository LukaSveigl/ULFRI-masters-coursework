# Description: Implementation of the optical flow methods for the first project of the 
# Advanced computer vision methods course. The methods are implemented as functions
# lucas_kanade and horn_schunck. The methods are called from the run_vaja_1.py file.

import numpy as np
import scipy.signal as sps
from typing import Tuple

from ex1_utils import gaussderiv, gausssmooth


def normalize_image(im: np.ndarray) -> np.ndarray:
    """
    Normalize the input image to the range [0, 1] and converts it to the float data type.

    Args:
        im: numpy.ndarray
            Input image.

    Returns:
        im_norm: numpy.ndarray
            Normalized image.
    """
    im_norm = im.astype(np.float32) / 255
    return im_norm


def lucas_kanade(im1: np.ndarray, im2: np.ndarray, N: int, sigma: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the basic Lucas-Kanade optical flow method. The method is implemented in multiple
    steps:
        1. Compute the spatial derivatives (x - horizontal, y - vertical) of both images.
        2. Compute the temporal derivative (t) of both images.
        3. Compute the neccessary products of the spatial derivatives.
        4. Compute the optical flow using the least squares method.

    Additionally, some improvements have been implemented:
        1. Smoothing of the temporal derivative (it) with a Gaussian filter.
        2. Averaging of the spatial derivatives (i1x, i1y, i2x, i2y) in frame t and t+1.

    Args:
        im1: numpy.ndarray
            First image.
        im2: numpy.ndarray
            Second image.
        N: int
            Size of the window used for computing the optical flow (N x N).
        sigma: float
            Standard deviation of the Gaussian filter used for smoothing the images.

    Returns:
        u: numpy.ndarray
            Horizontal component of the optical flow.
        v: numpy.ndarray
            Vertical component of the optical flow.
    """
    im1 = normalize_image(im1)
    im2 = normalize_image(im2)

    # Compute the spatial derivatives of both images.
    i1x, i1y = gaussderiv(im1, sigma)
    i2x, i2y = gaussderiv(im2, sigma)

    # Compute and smoothen the temporal derivative of both images.
    it = im2 - im1
    it = gausssmooth(it, sigma)

    # Calculate the average of the spatial derivatives.
    i1x = (i1x + i2x) / 2
    i1y = (i1y + i2y) / 2

    # Compute the neccessary products of the spatial derivatives.
    i1x2 = np.multiply(i1x, i1x)
    i1y2 = np.multiply(i1y, i1y)
    i1xy = np.multiply(i1x, i1y)
    i1xt = np.multiply(i1x, it)
    i1yt = np.multiply(i1y, it)

    # Compute the optical flow using the least squares method.
    kernel = np.ones((N, N))

    i1x2_conv = sps.convolve2d(i1x2, kernel, mode="same", boundary="symm")
    i1y2_conv = sps.convolve2d(i1y2, kernel, mode="same", boundary="symm")
    i1xy_conv = sps.convolve2d(i1xy, kernel, mode="same", boundary="symm")
    i1xt_conv = sps.convolve2d(i1xt, kernel, mode="same", boundary="symm")
    i1yt_conv = sps.convolve2d(i1yt, kernel, mode="same", boundary="symm")

    D = np.subtract(np.multiply(i1x2_conv, i1y2_conv), np.multiply(i1xy_conv, i1xy_conv))
    u = np.multiply(np.divide(np.subtract(np.multiply(i1y2_conv, i1xt_conv), np.multiply(i1xy_conv, i1yt_conv)), D + 1e-15), -1)
    v = np.multiply(np.divide(np.subtract(np.multiply(i1x2_conv, i1yt_conv), np.multiply(i1xy_conv, i1xt_conv)), D + 1e-15), -1)

    return u, v


def lucas_kanade_harris(im1: np.ndarray, im2: np.ndarray, N: int, k: float, sigma: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Lucas-Kanade optical flow method with the Harris corner detector. The method is
    implemented in multiple steps:
        1. Compute the spatial derivatives (x - horizontal, y - vertical) of both images.
        2. Compute the temporal derivative (t) of both images.
        3. Compute the neccessary products of the spatial derivatives.
        4. Compute the Harris corner detector response.
        5. Compute the optical flow using the least squares method.

    Additionally, some improvements have been implemented:
        1. Smoothing of the temporal derivative (it) with a Gaussian filter.
        2. Averaging of the spatial derivatives (i1x, i1y, i2x, i2y) in frame t and t+1.

    Args:
        im1: numpy.ndarray
            First image.
        im2: numpy.ndarray
            Second image.
        N: int
            Size of the window used for computing the optical flow.
        k: float
            Harris corner detector parameter.
        sigma: float
            Standard deviation of the Gaussian filter used for smoothing the images.

    Returns:
        u: numpy.ndarray
            Horizontal component of the optical flow.
        v: numpy.ndarray
            Vertical component of the optical flow.
    """
    im1 = normalize_image(im1)
    im2 = normalize_image(im2)

    # Compute the spatial derivatives of both images.
    i1x, i1y = gaussderiv(im1, sigma)
    i2x, i2y = gaussderiv(im2, sigma)

    # Compute and smoothen the temporal derivative of both images.
    it = im2 - im1
    it = gausssmooth(it, sigma)

    # Calculate the average of the spatial derivatives.
    i1x = (i1x + i2x) / 2
    i1y = (i1y + i2y) / 2

    # Compute the neccessary products of the spatial derivatives.
    i1x2 = np.multiply(i1x, i1x)
    i1y2 = np.multiply(i1y, i1y)
    i1xy = np.multiply(i1x, i1y)
    i1xt = np.multiply(i1x, it)
    i1yt = np.multiply(i1y, it)

    # Compute the Harris corner detector response.
    i1x2_conv = sps.convolve2d(i1x2, np.ones((N, N)), mode="same")
    i1y2_conv = sps.convolve2d(i1y2, np.ones((N, N)), mode="same")
    i1xy_conv = sps.convolve2d(i1xy, np.ones((N, N)), mode="same")

    D = np.multiply(i1x2_conv, i1y2_conv) - np.multiply(i1xy_conv, i1xy_conv)
    trace = i1x2_conv + i1y2_conv
    R = D - k * np.multiply(trace, trace)

    # Set the Harris corner detector threshold.
    t = 0.01 * np.max(R)

    # Compute the optical flow using the least squares method.
    u = np.multiply(np.divide(np.subtract(np.multiply(i1y2_conv, i1xt), np.multiply(i1xy_conv, i1yt)), D + 1e-10), -1)
    v = np.multiply(np.divide(np.subtract(np.multiply(i1x2_conv, i1yt), np.multiply(i1xy_conv, i1xt)), D + 1e-10), -1)

    # Set the optical flow to zero for non-corner pixels.
    u[R < t] = 0
    v[R < t] = 0

    return u, v


def lucas_kanade_wrapper(im1: np.ndarray, im2: np.ndarray, N: int, method: str='basic', k: float=0.04, sigma: float=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function for the Lucas-Kanade optical flow method. The function calls the appropriate Lucas-Kanade
    implementation based on the method parameter:
        - 'basic': basic Lucas-Kanade method (ignores the k and n_levels parameters)
        - 'harris': Lucas-Kanade method with the Harris corner detector (ignores the n_levels parameter)

    Args:
        im1: numpy.ndarray
            First image.
        im2: numpy.ndarray
            Second image.
        N: int
            Size of the window used for computing the optical flow.
        method: str
            Method for computing the optical flow.
        k: float
            Harris corner detector parameter.
        sigma: float
            Standard deviation of the Gaussian filter used for smoothing the images.

    Returns:
        u: numpy.ndarray
            Horizontal component of the optical flow.
        v: numpy.ndarray
            Vertical component of the optical flow.
    """
    if method not in {'basic', 'harris'}:
        raise ValueError(f"Unknown method: {method}. One of 'basic', 'harris' expected.")
    
    if method == 'basic':
        return lucas_kanade(im1, im2, N, sigma=sigma)
    elif method == 'harris':
        return lucas_kanade_harris(im1, im2, N, k, sigma=sigma)
    

def horn_schunck(im1: np.ndarray, im2: np.ndarray, n_iter: int, lmbda: float, return_iterations: bool, sigma: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Horn-Schunck optical flow method.

    Args:
        im1: numpy.ndarray
            First image.
        im2: numpy.ndarray
            Second image.
        n_iter: int
            Number of iterations.
        lmbda: float
            Regularization parameter.
        return_iterations: bool
            If True, the function returns the number of iterations needed for convergence.
        sigma: float
            Standard deviation of the Gaussian filter used for smoothing the images.

    Returns:
        u: numpy.ndarray
            Horizontal component of the optical flow.
        v: numpy.ndarray
            Vertical component of the optical flow.
    """
    im1 = normalize_image(im1)
    im2 = normalize_image(im2)
    
    im1 = gausssmooth(im1, sigma)
    im2 = gausssmooth(im2, sigma)

    x_kernel = np.array([[-1/2, 1/2], [-1/2, 1/2]], dtype=np.float32)
    y_kernel = np.array([[-1/2, -1/2], [1/2, 1/2]], dtype=np.float32)
    t_kernel = np.array([[1/4, 1/4], [1/4, 1/4]], dtype=np.float32)
    residual_lap_kernel = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]], dtype=np.float32)

    i1x = sps.convolve2d(im1, x_kernel, mode="same")
    i1y = sps.convolve2d(im1, y_kernel, mode="same")
    i2x = sps.convolve2d(im2, x_kernel, mode="same")
    i2y = sps.convolve2d(im2, y_kernel, mode="same")
    it = sps.convolve2d(im2 - im1, t_kernel, mode="same")

    # Calculate the average of the spatial derivatives.
    i1x = (i1x + i2x) / 2
    i1y = (i1y + i2y) / 2

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    D = lmbda + np.add(np.power(i1x, 2), np.power(i1y, 2))

    iter_counter = 0

    while True:
        # Compute the convolutions of the residuals. Using boundary="symm" improves the results massively.
        u_conv = sps.convolve2d(u, residual_lap_kernel, mode="same", boundary="symm")
        v_conv = sps.convolve2d(v, residual_lap_kernel, mode="same", boundary="symm")

        P = np.add(np.add(np.multiply(i1x, u_conv), np.multiply(i1y, v_conv)), it)

        u_new = np.subtract(u_conv, np.multiply(i1x, np.divide(P, D + 1e-10)))
        v_new = np.subtract(v_conv, np.multiply(i1y, np.divide(P, D + 1e-10)))

        # Check for convergence.
        if np.abs(u - u_new).max() < 0.005 and np.abs(v - v_new).max() < 0.005:
            break

        u = u_new
        v = v_new

        iter_counter += 1

    if return_iterations:
        return u, v, iter_counter
    return u, v


def horn_schunk_lk(im1: np.ndarray, im2: np.ndarray, n_iter: int, lmbda: float, N: int, return_iterations: bool, sigma: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Horn-Schunk optical flow method initialized with the Lucas-Kanade method.

    Args:
        im1: numpy.ndarray
            First image.
        im2: numpy.ndarray
            Second image.
        n_iter: int
            Number of iterations.
        lmbda: float
            Regularization parameter.
        N: int
            Size of the window used for computing the optical flow using the Lucas-Kanade method.
        return_iterations: bool
            If True, the function returns the number of iterations needed for convergence.
        sigma: float
            Standard deviation of the Gaussian filter used for smoothing the images.

    Returns:
        u: numpy.ndarray
            Horizontal component of the optical flow.
        v: numpy.ndarray
            Vertical component of the optical flow.
    """
    im1 = normalize_image(im1)
    im2 = normalize_image(im2)
    
    im1 = gausssmooth(im1, sigma)
    im2 = gausssmooth(im2, sigma)

    x_kernel = np.array([[-1/2, 1/2], [-1/2, 1/2]], dtype=np.float32)
    y_kernel = np.array([[-1/2, -1/2], [1/2, 1/2]], dtype=np.float32)
    t_kernel = np.array([[1/4, 1/4], [1/4, 1/4]], dtype=np.float32)
    residual_lap_kernel = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]], dtype=np.float32)

    i1x = sps.convolve2d(im1, x_kernel, mode="same")
    i1y = sps.convolve2d(im1, y_kernel, mode="same")
    i2x = sps.convolve2d(im2, x_kernel, mode="same")
    i2y = sps.convolve2d(im2, y_kernel, mode="same")
    it = sps.convolve2d(im2 - im1, t_kernel, mode="same")

    # Calculate the average of the spatial derivatives.
    i1x = (i1x + i2x) / 2
    i1y = (i1y + i2y) / 2

    u, v = lucas_kanade(im1, im2, N)

    D = lmbda + np.add(np.power(i1x, 2), np.power(i1y, 2))

    iter_counter = 0

    # Run the Horn-Schunck method until convergence.
    while True:
        # Compute the convolutions of the residuals. Using boundary="symm" improves the results massively.
        u_conv = sps.convolve2d(u, residual_lap_kernel, mode="same", boundary="symm")
        v_conv = sps.convolve2d(v, residual_lap_kernel, mode="same", boundary="symm")

        P = np.add(np.add(np.multiply(i1x, u_conv), np.multiply(i1y, v_conv)), it)

        u_new = np.subtract(u_conv, np.multiply(i1x, np.divide(P, D + 1e-10)))
        v_new = np.subtract(v_conv, np.multiply(i1y, np.divide(P, D + 1e-10)))

        # Check for convergence.
        #if np.all(np.abs(u - u_new) < 0.005) and np.all(np.abs(v - v_new) < 0.005):
        if np.abs(u - u_new).max() < 0.005 and np.abs(v - v_new).max() < 0.005:
            break

        u = u_new
        v = v_new

        iter_counter += 1

    if return_iterations:
        return u, v, iter_counter
    return u, v


def horn_schunk_wrapper(im1: np.ndarray, im2: np.ndarray, n_iter: int, lmbda: float, method: str='basic', N: int = 3, sigma: float = 1, return_iterations: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function for the Horn-Schunck optical flow method. The function calls the appropriate Horn-Schunck
    implementation based on the method parameter:
        - 'basic': basic Horn-Schunck method
        - 'lk': Horn-Schunck method initialized with the Lucas-Kanade method

    Args:
        im1: numpy.ndarray
            First image.
        im2: numpy.ndarray
            Second image.
        n_iter: int
            Number of iterations.
        lmbda: float
            Regularization parameter.
        method: str
            Method for computing the optical flow.
        N: int
            Size of the window used for computing the optical flow using the Lucas-Kanade method.
        sigma: float
            Standard deviation of the Gaussian filter used for smoothing the images.
        return_iterations: bool
            If True, the function returns the number of iterations needed for convergence.

    Returns:
        u: numpy.ndarray
            Horizontal component of the optical flow.
        v: numpy.ndarray
            Vertical component of the optical flow.
    """
    if method not in {'basic', 'lk'}:
        raise ValueError(f"Unknown method: {method}. One of 'basic', 'lk' expected.")
    
    if method == 'basic':
        return horn_schunck(im1, im2, n_iter, lmbda, return_iterations, sigma=sigma)
    elif method == 'lk':
        return horn_schunk_lk(im1, im2, n_iter, lmbda, N, return_iterations, sigma=sigma)
