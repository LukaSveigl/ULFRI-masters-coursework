import argparse
import os
import cv2

import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
#from siamfc import TrackerSiamFC
from lt_siamfc import TrackerSiamFC

def evaluate_tracker(dataset_path, network_path, results_dir, visualize):
    
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    tracker = TrackerSiamFC(net_path=network_path)

    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):

            img = cv2.imread(sequence.frame(i))
            prediction, score, bboxes = tracker.update(img)

            #prediction, score = tracker.update(img)
            results.append(prediction)
            scores.append([score])

            if visualize:
                if not np.isnan(prediction[0]):
                    tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                    br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                    cv2.rectangle(img, tl_, br_, (0, 0, 255), 2)

                # Draw ground-truth
                gt_rect = sequence.get_annotation(i)
                tl_ = (int(round(gt_rect[0])), int(round(gt_rect[1])))
                br_ = (int(round(gt_rect[0] + gt_rect[2])), int(round(gt_rect[1] + gt_rect[3])))
                cv2.rectangle(img, tl_, br_, (0, 255, 0), 2)

                # for bbox in bboxes:
                #     tl_ = (int(round(bbox[0])), int(round(bbox[1])))
                #     br_ = (int(round(bbox[0] + bbox[2])), int(round(bbox[1] + bbox[3])))
                #     cv2.rectangle(img, tl_, br_, (255, 0, 0), 2)
# 
                # if len(bboxes) > 0:
                #     # Save the image as pdf.
                #     # If image already exists, do not save it again.
                #     if not os.path.exists('gaussian_sampled_images.png'):
                #         cv2.imwrite('gaussian_sampled_images.png', img)
                #         saved_frame = i
                #     else:
                #         if saved_frame + 5 == current_frame:
                #             cv2.imwrite(f'gaussian_sampled_images_{current_frame}.png', img)
                #             saved_frame = 0

                if len(scores) > 0 and scores[-1][0] > 4 and scores[-1][0] < 4.5:
                    # Before object is lost - save the image
                    if not os.path.exists(f'before_lost_object_{sequence_name}_{i}.png'):
                        cv2.imwrite(f'before_lost_object_{sequence_name}_{i}.png', img)

                if len(scores) > 0 and scores[-1][0] < 4:
                    # Object is lost - save the image
                    if not os.path.exists(f'lost_object_{sequence_name}_{i}.png'):
                        cv2.imwrite(f'lost_object_{sequence_name}_{i}.png', img)

                if len(scores) > 0 and scores[-1][0] > 4.5 and scores[-2][0] < 4.5:
                    # Object is found - save the image
                    if not os.path.exists(f'found_object_{sequence_name}_{i}.png'):
                        cv2.imwrite(f'found_object_{sequence_name}_{i}.png', img)

                cv2.imshow('win', img)
                key_ = cv2.waitKey(10)
                if key_ == 27:
                    exit(0)
        
        save_results(results, bboxes_path)
        save_results(scores, scores_path)


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')

args = parser.parse_args()

evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize)
