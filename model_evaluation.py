import os
from os import path

import sys
import argparse
import json
import collections

MIN_LOCALIZATION_IOU = 0.5

parser = argparse.ArgumentParser(description="Evaluatiing models performanc.")

parser.add_argument("--groudtruth", default="eval_data/sample_testset/out_res.json",
                    help="json files containing ground truth value")
parser.add_argument("--predictions", default="predictions.json",
                    help="json files containing prediction value")

parser.add_argument("--root-dir", default=os.curdir,
                    help="root folder")


def avg_precision(expected, predictions):
    k = min(len(expected), len(predictions))
    score = 0.0
    correct_items = 0.0
    for idx, predicted in enumerate(predictions):
        if predicted in expected:
            correct_items += 1.0
            score += (correct_items / (idx+1))
    return score/k

def intersection_over_union(expected, exp_bboxes, predictions, pred_bboxes):

    def iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        intersect_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
        union_area = area1 + area2 - intersect_area
        return intersect_area / union_area

    correct_items = 0
    ious = []
    for i, expect in enumerate(expected):
        correct_prediction = False
        for j, predicted in enumerate(predictions):
            if predicted == expect:
                correct_items += 1
                ious.append(iou(exp_bboxes[i], pred_bboxes[j]))
                correct_prediction = True
                break
        if not correct_prediction:
            ious.append(0.0)
    return ious, correct_items

def intersection_over_union(expected, exp_bboxes, predictions, pred_bboxes):
    def iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        intersect_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
        union_area = area1 + area2 - intersect_area
        return intersect_area / union_area

    correct_items = 0
    ious = []
    for i, expect in enumerate(expected):
        correct_prediction = False
        for j, predicted in enumerate(predictions):
            if predicted == expect:
                correct_items += 1
                ious.append(iou(exp_bboxes[i], pred_bboxes[j]))
                correct_prediction = True
                break
        if not correct_prediction:
            ious.append(0.0)
    return ious, correct_items


def main(args):

    args = parser.parse_args(args)

    with open(args.predictions) as f:
        predictions_data = json.load(f)

    with open(args.groudtruth) as f:
        groundtruth_data = json.load(f)

    retrieval_precision_sum = 0.0
    localization_precision_sum = 0.0
    iou_sum = 0.0
    correct_items = 0
    total_items = 0

    for image, crops in predictions_data.items():
        predicted_items = [c[0] for c in crops]

        if image not in groundtruth_data:
            continue
        groundtruth_crops = groundtruth_data[image]

        grountruth_items = [c[0] for c in groundtruth_crops]
        predicted_items_boxes = [c[1] for c in crops]
        grountruth_items_boxes = [c[1] for c in groundtruth_crops]

        print(grountruth_items, predicted_items)
        retrieval_precision_sum += avg_precision(grountruth_items, predicted_items)

        ious, n_correct = intersection_over_union(grountruth_items, grountruth_items_boxes,
                                                  predicted_items, grountruth_items_boxes)
        iou_sum += sum(ious)
        correct_items += n_correct
        total_items += len(ious)

        for iou, exp in zip(ious, grountruth_items):
            if iou < MIN_LOCALIZATION_IOU:
                if exp in predicted_items:
                    # Remove all badly localized items
                    predicted_items[predicted_items.index(exp)] = ''
        localization_precision_sum += avg_precision(grountruth_items, predicted_items)

    retrieval_map = retrieval_precision_sum / len(predictions_data)
    localization_map = localization_precision_sum / len(predictions_data)
    print('Retrieval performance:')
    print('\tmAP : {:.4f}'.format(retrieval_map))
    print('\tlocalization mAP : {:.4f}'.format(localization_map))

if __name__ == '__main__':
    main(sys.argv[1:])

