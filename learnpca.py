#!/usr/bin/env python3
import argparse
import json
import os
import sys
from os.path import join, exists

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))

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
parser.add_argument('--root-dir', default=os.curdir,
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

    for idx, image_name in enumerate(images):
        print('{}/{}: extracting features of image {}'.format(idx+1,
                                                              len(images),
                                                              image_name))
        image_path = join(image_dir, image_name)
        image = loadImage(image_path)

        features = compute_features(model, image)

        features_data.append(features)

    return features_data


def learn_pca(features_data, name, features_dir):
    """Computes regional mac features and learns PCA on them to be able
    to whiten them later.
    This method assumes that there is enough memory available to hold all
    r_macs and do the PCA computation. If this assumption does not hold,
    we can switch to a memmapped numpy array and IncrementalPCA.
    """
    r_macs = []
    for feature in features_data:
        r_macs += compute_r_macs(feature)

    r_macs = np.vstack(r_macs)


    print('Extracted {} rmacs on {} images'.format(r_macs.shape[0],
                                                   len(features_data)))
    print("total rmacs shape {}".format(r_macs.shape))

    pca = PCA(n_components=r_macs.shape[1], whiten=True)
    pca.fit(r_macs)

    pca_path = join(features_dir, '{}.pca'.format(name))

    joblib.dump(pca, pca_path)

    print('Computed PCA and saved it to {}'.format(pca_path))

def main(args):
    args = parser.parse_args(args)

    root_dir = args.root_dir

    name = args.name
    if args.root_dir:
        root_dir = os.path.abspath(args.root_dir)


    if not exists(args.features_dir):
        os.mkdir(args.features_dir)


    features_data = extract_conv_features(name , args.features_dir,args.image_dir, root_dir)

    learn_pca(features_data, name, args.features_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
