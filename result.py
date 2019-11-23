import os
import sys
import argparse
from timeit import default_timer as timer
from os.path import join, basename
from os.path import join, exists
from glob import glob

import json

from src.util import loadImage
from src.search import SearchModel, search

parser = argparse.ArgumentParser(description=
                                 'Query an image for similar images')

parser.add_argument('--features', default='features',
                    help='Path to features to use')

parser.add_argument('--image-dir', default="data/query",
                    help='image directories containing query images')

parser.add_argument('--output-file', default="output.json",
                    help="name of the output json file. default: output.json")


def main(args):
    import numpy as np

    args = parser.parse_args(args)

    if not exists(args.image_dir):
        print("Path {} does not exist".format(args.image_dir))

    images_path = args.image_dir


    images = os.listdir(images_path)
    extensions = ['.png', '.jpg', '.jpeg']
    images = [img for img in images
              if os.path.splitext(img)[1].lower() in extensions]

    search_model = SearchModel(args.features )

    result_data = {}

    for image_file in images:

        if not exists(os.path.join(images_path, image_file)):
            print("{} does not exists. Skipping...".format(image_file))
            continue


        query = loadImage(join(images_path, image_file))

        results, similarities, bboxes = search(search_model, query, top_n=1)

        print('Top result for query image {}'.format(image_file))

        result_path = search_model.get_metadata(results[0])['image']

        image_file_name = image_file.split('.')[0]

        if result_path not in result_data:
            similar_images = [image_file_name]
            image_sect = list(bboxes[0])
            similar_images.append(image_sect)
            image_section = [similar_images]
            res_path = result_path.split('/')[-1].split('.')[0]
            result_data[res_path] = image_section

        else:
            res_path = result_path.split('/')[-1].split('.')[0]
            similar_images = [image_file_name]
            image_sect = list(bboxes[0])
            similar_images.append(image_sect)

            result_data[res_path].append(similar_images)


        print('{}\t{:.4f}\t{}'.format(result_path, similarities[0], bboxes[0]))


    meta_file_name = args.output_file
    with open(meta_file_name, 'w') as f:
        json.dump(result_data, f)

if __name__ == '__main__':
    main(sys.argv[1:])
