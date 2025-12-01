# This file contains the implementation of the truth generator. The truth generator is 
# used to extract the ground truths for an image from it's respective YOLO annotation.
# Using the ground truths, the generator generates cropped images of the detected object
# and full images with the rectangle around the detected object.

import common, utils
import json, cv2


def load_image_and_draw_rect(infile):
    """
    Loads the image and draws the rectangle around the detected object.
    Additionally, saves the cropped image without the rectangle, as it 
    will be used for benchmarking the LBP.

    :param infile: path to the input image
    """
    img = cv2.imread(infile)
    with open(infile.replace('.png', '.txt')) as f:
        # Read the YOLO coordinates and convert them to the OpenCV format.
        x_center, y_center, width, height = [float(x) for x in f.readline().split()][1::]
        x, y, w, h = utils.yolo_to_opencv(x_center, y_center, width, height, img.shape[1], img.shape[0])

        # Crop the image to the detected object and save it.
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(common.OUT_IMAGES_CR_TRUTHS + infile.split('/')[-1], crop_img)

        # Draw the rectangle around the detected object and save it.
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
        cv2.imwrite(common.OUT_IMAGES_TRUTHS + infile.split('/')[-1], img)


def main():
    """
    Main function.
    """
    # Load the annotations (is left or right ear in the image?) from the JSON file.
    annotations = json.load(open(common.ANNOTATIONS))

    # Loop through all images, draw the rect and write them to disk.
    for key in annotations.keys():
        load_image_and_draw_rect(common.SRC_IMAGES + key)


if __name__ == '__main__':
    print('Running truth_generator.py...')
    main()
    print('Done.')