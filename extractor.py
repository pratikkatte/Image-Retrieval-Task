import argparse
import json
import os
import sys
from os.path import join, exists

import numpy as np
from sklearn.externals import joblib

from src.util import loadImage
from src.model import loadModel
from src.features import compute_features, \
                         compute_r_macs, \
                         compute_representation


parser = argparse.ArgumentParser(description=
                                 'Extract feature representations')
parser.add_argument('--features-dir',
                    help='Folder where extracted data is stored',
                    default='features')
parser.add_argument('--root-dir', default=None,
                    help='Directory to which relative paths are taken')
parser.add_argument('--image-dir',
                    help='Folder where images are read from',
                    default='data/images')

parser.add_argument('--name', default='features',
                    help='(Dataset) name to use for extracted files')

def extract_conv_features(name, features_dir, image_dir, root_dir):
    """Extracts features of all images in image_dir and
    saves them for later use.
    """

    image_dir = os.path.abspath(image_dir)

    out_dir = join(features_dir, 'features/')
    if not exists(out_dir):
        os.mkdir(out_dir)

    extensions = ['.png', '.jpg', '.jpeg']
    images = os.listdir(image_dir)
    images = [img for img in images
              if os.path.splitext(img)[1].lower() in extensions]
    images = sorted(images)

    model = loadModel()

    meta_data = {}
    features_data = []

    for index, image_name in enumerate(images):
        print('{}/{}: extracting features of image {}'.format(index+1,
                                                              len(images),
                                                              image_name))
        image_path = join(image_dir, image_name)
        image = loadImage(image_path)

        features = compute_features(model, image)

        features_data.append(features)
        #
        np.save(join(out_dir, os.path.basename(image_name)), features)
        meta_data[index] = {
            'image': os.path.relpath(image_path, root_dir),
            'height': image.shape[0],
            'width': image.shape[1]
        }

    meta_file_name = '{}.meta'.format(name)
    with open(join(features_dir, meta_file_name), 'w') as f:
        json.dump(meta_data, f)

    return features_data

def compute_global_representation(features_data, name, features_dir, pca=None):
    """Uses previously extracted features to compute an image representation
    which is suitable for image retrieval.
    """
    repr_store = None
    for index, feature in enumerate(features_data):

        representation = compute_representation(feature, pca)

        if repr_store is None:
            repr_store = np.empty((len(features_data), representation.shape[-1]))

        repr_store[index] = np.squeeze(representation, axis=0)


    repr_file_path = join(features_dir, '{}.repr.npy'.format(name))

    np.save(repr_file_path, repr_store)
    print('Computed {} image representations and '
          'saved them to {}'.format(len(features_data), repr_file_path))


def main(args):
    args = parser.parse_args(args)


    if not exists(args.features_dir):
        os.mkdir(args.features_dir)

    features_data = extract_conv_features(args.name, args.features_dir, args.image_dir,args.root_dir)
    pca_path = join(args.features_dir, '{}.pca'.format(args.name))

    if exists(pca_path):
        pca = joblib.load(pca_path)

    compute_global_representation(features_data, args.name, args.features_dir, pca)

if __name__ == '__main__':
    main(sys.argv[1:])
