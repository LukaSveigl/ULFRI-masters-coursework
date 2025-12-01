# Description: This file contains the test implementation of the Mean Shift algorithm, which is used to asses
# the validity of the Mean Shift algorithm on pre-defined test matrices. In the experiment, multiple different
# starting positions, kernel shapes and sizes and termination conditions are tested. The results are visualized
# and exported in pdf format to the results/msmode directory.

import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List

from ex1_utils import gausssmooth
from ex2_utils import generate_responses_1, get_patch


def generate_responses_2() -> np.ndarray:
    """
    Generates a response matrix with 3 peaks.

    Returns:
        np.ndarray: The response matrix.
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    responses[80, 80] = 0.3
    return gausssmooth(responses, 10)


def generate_responses_3() -> np.ndarray:
    """
    Generates a response matrix with 4 peaks.

    Returns:
        np.ndarray: The response matrix.
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    responses[80, 80] = 0.7
    responses[50, 50] = 0.2
    return gausssmooth(responses, 10)


def generate_responses_4() -> np.ndarray:
    """
    Generates a response matrix with 5 peaks.

    Returns:
        np.ndarray: The response matrix.
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    responses[80, 80] = 0.3
    responses[50, 50] = 0.7
    responses[30, 30] = 0.6
    return gausssmooth(responses, 10)


def generate_responses_all() -> List[np.ndarray]:
    """
    Aggregates all response matrices into a list.

    Returns:
        List[np.ndarray]: The list of response matrices.
    """
    responses_list = []
    for i in range(1, 5):
        responses = globals()[f'generate_responses_{i}']()
        responses_list.append(responses)
    return responses_list


if __name__ == '__main__':
    responses_list = generate_responses_all()
    starts = [(65, 50), (30, 70), (80, 80), (50, 40)]
    kernel_shapes = [(10, 10, 1), (20, 20, 1)]
    termination_conditions = [0.5, 0.2]

    for index, responses in enumerate(responses_list):
        for start_x, start_y in starts:
            print(f'Start: ({start_x}, {start_y})')
            for kernel_x, kernel_y, sigma in kernel_shapes:
                print(f'+> Kernel: ({kernel_x}, {kernel_y}), Sigma: {sigma}')

                kernel_x_half = int(kernel_x / 2)
                kernel_y_half = int(kernel_y / 2)

                # Create a meshgrid for the kernel.
                X, Y = np.meshgrid(np.arange(-kernel_x_half, kernel_x_half), np.arange(-kernel_y_half, kernel_y_half))

                for threshold in termination_conditions:

                    # Skip the combinations that would take too long.
                    if index >= 2 and threshold < 0.5:
                        continue

                    print(f'--> Threshold: {threshold}')

                    updated_start_x = start_x
                    updated_start_y = start_y

                    iterations = 0

                    while True:
                        iterations += 1

                        patch, mask = get_patch(responses, (updated_start_x, updated_start_y), (kernel_x, kernel_y))
                        patch = patch.astype(np.float32)

                        # Calculate the weighted mean of the patch.
                        patch = patch / np.sum(patch)
                        x_change = np.sum(patch * X)
                        y_change = np.sum(patch * Y)

                        # Update the starting position.
                        updated_start_x += x_change
                        updated_start_y += y_change

                        
                        # Check if the calculated shift is smaller than the threshold.
                        if np.sqrt(x_change ** 2 + y_change ** 2) < threshold:
                            break

                    print(f'Start: ({start_x}, {start_y}), End: ({updated_start_x}, {updated_start_y}), Iterations: {iterations}')

                    plt.clf()
                    plt.imshow(responses)
                    plt.title(f'Start: ({start_x}, {start_y}), Kernel: ({kernel_x}, {kernel_y}), Threshold: {threshold}, Iterations: {iterations}')
                    plt.plot(start_x, start_y, 'ro')
                    plt.plot(updated_start_x, updated_start_y, 'bo')
                    plt.legend(['Start', 'End'])
                    plt.savefig(f'./results/msmode/responses_{index + 1}_{start_x}_{start_y}_{kernel_x}_{kernel_y}_{threshold}.pdf', format='pdf')
