# This file contains utilities used all throughout the project.

import numpy as np

def yolo_to_opencv(x_center: float, y_center: float, width: float, height: float, img_width: int, img_height: int) -> list:
    """
    Converts the yolo format to the opencv format.

    :param x_center: x coordinate of the center of the object.
    :param y_center: y coordinate of the center of the object.
    :param width: Width of the object.
    :param height: Height of the object.
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :returns: List of coordinates in the opencv format.
    """
    return [
        int((x_center - width / 2) * img_width),
        int((y_center - height / 2) * img_height),
        int(width * img_width),
        int(height * img_height)
    ]


def opencv_to_yolo(x: int, y: int, w: int, h: int, img_width: int, img_height: int) -> list:
    """
    Converts the opencv format to the yolo format.

    :param x: x coordinate of the center of the object.
    :param y: y coordinate of the center of the object.
    :param w: Width of the object.
    :param h: Height of the object.
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :returns: List of coordinates in the yolo format.
    """
    return [
        (x + w / 2) / img_width,
        (y + h / 2) / img_height,
        w / img_width,
        h / img_height
    ]


def iou(truth: list, detection: list) -> float:
    """
    Calculates the intersection over union (IoU) of the two rectangles.

    :param truth: Ground truth rectangle.
    :param detection: Detected rectangle.
    :returns: IoU of the two rectangles.
    """
    # Calculate the intersection rectangle.
    x1 = max(truth[0], detection[0])
    y1 = max(truth[1], detection[1])
    x2 = min(truth[0] + truth[2], detection[0] + detection[2])
    y2 = min(truth[1] + truth[3], detection[1] + detection[3])

    # Calculate the area of the intersection rectangle.
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both rectangles.
    truth_area = (truth[2] + 1) * (truth[3] + 1)
    detection_area = (detection[2] + 1) * (detection[3] + 1)

    # Calculate the union area.
    union_area = truth_area + detection_area - intersection_area

    # Calculate the intersection over union.
    iou = intersection_area / float(union_area)

    return iou


def image_score(detections: list, ground_truths: list, threshold: float) -> float:
    """
    Calculates the score of the image.

    :param detections: List of detected objects.
    :param ground_truths: List of ground truths.
    :param threshold: Threshold for the IoU.
    :returns: Score of the image.
    """
    image_results = {'TP': 0, 'FP': 0, 'FN': 0}

    # Compute the IoU for each detection.
    for detection in detections:
        image_iou = iou(ground_truths, detection)
        # If the IoU is greater than the threshold, it is a true positive.
        if image_iou > threshold:
            # There can only be one true positive, as each image contains only one ear.
            if image_results['TP'] == 0:
                image_results['TP'] += 1
            else:
                image_results['FP'] += 1
        else:
            image_results['FP'] += 1
    if len(detections) == 0:
        image_results['FN'] += 1
    return image_results, image_results['TP'] / (image_results['TP'] + image_results['FP'] + image_results['FN'])


def print_histogram_stats(hist: np.ndarray) -> None:
    """
    Prints the statistics of the given histogram.

    :param hist: Histogram to print the statistics for.
    """
    print('Histogram statistics:')
    print('Mean: {}'.format(np.mean(hist)))
    print('Median: {}'.format(np.median(hist)))
    print('Std: {}'.format(np.std(hist)))
    print('Min: {}'.format(np.min(hist)))
    print('Max: {}'.format(np.max(hist)))
    print('Sum: {}'.format(np.sum(hist)))
    print('Variance: {}'.format(np.var(hist)))
    print('Size: {}'.format(len(hist)))
