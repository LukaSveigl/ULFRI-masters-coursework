import cv2
import time

import numpy as np
import matplotlib.pyplot as plt

from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade_wrapper, horn_schunk_wrapper

# Lucas-Kanade and Horn-Schunck parameters.
kernel_size = 10
k = 0.04

n_iter = 1000
alpha = 0.5

sigma = 1

def plot_results(im1: np.ndarray, im2: np.ndarray, u: np.ndarray, v: np.ndarray, title: str):
    """
    Plots the results of the optical flow methods.

    Args:
        im1: np.ndarray
            First image.
        im2: np.ndarray
            Second image.
        u: np.ndarray
            Horizontal flow.
        v: np.ndarray
            Vertical flow.
        title: str
            Title of the plot.
    """
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(u, v, ax1_21, type='angle')
    show_flow(u, v, ax1_22, type='field' , set_aspect=True)
    fig1.suptitle(title)
    plt.show()


def test_noise():
    """
    Tests the basic implementations of the optical flow methods on a pair of noisy images,
    displaying the results.
    """
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='basic')
    U_hs, V_hs = horn_schunk_wrapper(im1, im2, n_iter, 0.5, method='basic')

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade basic implementation - noise')
    plot_results(im1, im2, U_hs, V_hs, 'Horn-Schunck basic implementation - noise')


def test_basic_on_pairs():
    """
    Tests the basic implementations of the optical flow methods on 3 pairs of images, 
    displaying the results.
    """
    # Pair of images from the collision dataset.
    im1 = cv2.imread('collision/00000173.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('collision/00000174.jpg', cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='basic')
    U_hs, V_hs = horn_schunk_wrapper(im1, im2, n_iter, alpha, method='basic')

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade basic implementation - collision')
    plot_results(im1, im2, U_hs, V_hs, 'Horn-Schunck basic implementation - collision')

    # Pair of images from the lab2 dataset.
    im1 = cv2.imread('lab2/079.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('lab2/080.jpg', cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='basic')
    U_hs, V_hs = horn_schunk_wrapper(im1, im2, n_iter, alpha,  method='basic')

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade basic implementation - lab2')
    plot_results(im1, im2, U_hs, V_hs, 'Horn-Schunck basic implementation - lab2')

    # Pair of images from the disparity dataset.
    im1 = cv2.imread('disparity/cporta_left.png', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('disparity/cporta_right.png', cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='basic')
    U_hs, V_hs = horn_schunk_wrapper(im1, im2, n_iter, alpha, method='basic')

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade basic implementation - disparity')
    plot_results(im1, im2, U_hs, V_hs, 'Horn-Schunck basic implementation - disparity')


def harris_lucas_kanade():
    """
    Tests the Lucas-Kanade method with Harris corner detection on a pair of images,
    displaying the results.
    """
    # Pair of images from the collision dataset.
    im1 = cv2.imread('collision/00000173.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('collision/00000174.jpg', cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='harris', k=k)

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade with Harris corner detection - collision')

    # Pair of images from the lab2 dataset.
    im1 = cv2.imread('lab2/079.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('lab2/080.jpg', cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='harris', k=k)

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade with Harris corner detection - lab2')

    # Pair of images from the disparity dataset.
    im1 = cv2.imread('disparity/cporta_left.png', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('disparity/cporta_right.png', cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='harris', k=k)

    plot_results(im1, im2, U_lk, V_lk, 'Lucas-Kanade with Harris corner detection - disparity')


def parameters_test():
    """
    Tests both methods with different combinations of parameters on the 'collision' dataset.
    """
    im1 = cv2.imread('collision/00000173.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('collision/00000174.jpg', cv2.IMREAD_GRAYSCALE)

    # Test different kernel sizes and sigmas for the Lucas-Kanade method.
    for kernel_size in [5, 10, 15]:
        for sigma in [0.5, 1, 1.5]:
            U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='basic', sigma=sigma)
            plot_results(im1, im2, U_lk, V_lk, f'Lucas-Kanade basic implementation - collision, kernel size {kernel_size}, sigma {sigma}')

    # Test different alpha values and sigmas for the Horn-Schunck method.
    for alpha in [0.1, 0.5, 1]:
        for sigma in [0.5, 1, 1.5]:
            U_hs, V_hs = horn_schunk_wrapper(im1, im2, n_iter, alpha, method='basic', sigma=sigma)
            plot_results(im1, im2, U_hs, V_hs, f'Horn-Schunck basic implementation - collision, alpha {alpha}, sigma {sigma}')


def time_measurements():
    """
    Measures the time it takes (and number of iterations in the Horn-Schunk method) to compute 
    the optical flow for a pair of images using the Lucas-Kanade and Horn-Schunck methods. The
    time is measured in milliseconds.
    """
    image_pairs = [
        ('collision/00000173.jpg', 'collision/00000174.jpg'),
        ('lab2/079.jpg', 'lab2/080.jpg'),
        ('disparity/cporta_left.png', 'disparity/cporta_right.png')
    ]

    # Perform the measurements 10 times and print the results. 
    for pair in image_pairs:
        lk_basic_times = []
        hs_basic_times = []
        hs_basic_iterations = []
        lk_harris_times = []
        hs_lk_times = []
        hs_lk_iterations = []

        name = pair[0].split('/')[0]

        for i in range(10):
            print(f'Iteration {i + 1}')
            im1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)

            start = time.time()
            U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='basic')
            end = time.time()
            time_ms = (end - start) * 1000
            lk_basic_times.append(time_ms)
            print(f'    Lucas-Kanade basic implementation - {name}: {time_ms} milliseconds')

            start = time.time()
            U_hs, V_hs, iterations = horn_schunk_wrapper(im1, im2, n_iter, alpha, method='basic', return_iterations=True)
            end = time.time()
            time_ms = (end - start) * 1000
            hs_basic_times.append(time_ms)
            hs_basic_iterations.append(iterations)
            print(f'    Horn-Schunck basic implementation - {name}: {time_ms} milliseconds, {iterations} iterations')

            start = time.time()
            U_lk, V_lk = lucas_kanade_wrapper(im1, im2, kernel_size, method='harris', k=k)
            end = time.time()
            time_ms = (end - start) * 1000
            lk_harris_times.append(time_ms)
            print(f'    Lucas-Kanade with Harris corner detection - {name}: {time_ms} milliseconds')

            start = time.time()
            U_hs, V_hs, iterations = horn_schunk_wrapper(im1, im2, n_iter, alpha, method='lk', N=kernel_size, return_iterations=True)
            end = time.time()
            time_ms = (end - start) * 1000
            hs_lk_times.append(time_ms)
            hs_lk_iterations.append(iterations)
            print(f'    Horn-Schunck with Lucas-Kanade - {name}: {time_ms} milliseconds, {iterations} iterations')

            print('')

        print('Average times:')
        print(f'    Lucas-Kanade basic implementation - {name}: Average {np.mean(lk_basic_times)} milliseconds')
        print(f'    Horn-Schunck basic implementation - {name}: Average {np.mean(hs_basic_times)} milliseconds, {np.mean(hs_basic_iterations)} iterations')
        print(f'    Lucas-Kanade with Harris corner detection - {name}: Average {np.mean(lk_harris_times)} milliseconds')
        print(f'    Horn-Schunck with Lucas-Kanade - {name}: Average {np.mean(hs_lk_times)} milliseconds, {np.mean(hs_lk_iterations)} iterations')


if __name__ == '__main__':
    # test_noise()
    # test_basic_on_pairs()
    # harris_lucas_kanade()
    parameters_test()
    # time_measurements()
