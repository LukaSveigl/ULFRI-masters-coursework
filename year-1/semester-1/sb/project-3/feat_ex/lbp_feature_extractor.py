import torch, os, sys, cv2, PIL, numpy as np
import torchvision.models
from ultralytics import YOLO
import os
from torchvision import transforms
from lbp import cLBP


if __name__ == "__main__":
    print("\n")
    folder_dir = '../datasets/ears/images/val'
    features_list = []

    for filename in os.listdir(folder_dir):
        print(filename)
        # image = PIL.Image.open(os.path.join(folder_dir, filename)).convert("RGB")
        #image = cv2.imread(os.path.join(folder_dir, filename))

        # Resize image
        #image = cv2.resize(image, (128, 128))
        image = cLBP.load_image(os.path.join(folder_dir, filename))
        image = cv2.resize(image, (128, 128))

        features, _ = cLBP.compute_lbp(image, region_dimensions=(16, 16), P=8, R=1)
        features_list.append(features)

        filename = filename.replace('images', 'features-lbp')
        filename = filename.replace('.png', '.txt')

        features_folder_dir = folder_dir.replace('images', 'features-lbp')

        # Flatten features-resnet
        # features-resnet = features-resnet.flatten()

        np.savetxt(os.path.join(features_folder_dir, filename), features, delimiter=',')

        del features

    print(features_list)
