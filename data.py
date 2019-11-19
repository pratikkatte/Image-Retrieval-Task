import argparse
import urllib.request
import os
import sys


parser = argparse.ArgumentParser(description="download Dataset")

parser.add_argument("--root-dir", default=os.curdir,
                    help='relative path directory'
                    )


def main(args):
    args = parser.parse_args(args)

    if args.root_dir:
        root_dir = os.path.abspath(args.root_dir)

    if not os.path.exists(os.path.join(root_dir, "data")):
        os.mkdir(os.path.join(root_dir, 'data'))

    if not os.path.isfile(os.path.join(root_dir, 'images.txt')):
        print("downloading the image dataset from https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/images.txt")
        urllib.request.urlretrieve('https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/images.txt', 'images.txt')

    if not os.path.isfile(os.path.join(root_dir, 'crops.txt')):
        print("Download the query dataset from https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/crops.txt")
        urllib.request.urlretrieve('https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/crops.txt','crops.txt')

    with open(os.path.join(root_dir, 'images.txt')) as f:
        images = f.readlines()

    os.chdir(root_dir)

    if not os.path.exists(os.path.join('data', 'images')):
        os.mkdir(os.path.join('data', 'images'))
        print("created images directory in data")

    print("Downloading images")
    for image in images:
        image_name = image.split('/')[-1].split('.')[0]
        image_path = 'data/images/'+image_name+'.jpg'
        urllib.request.urlretrieve(image, os.path.join(root_dir,image_path))


    if not os.path.exists(os.path.join('data', 'query')):
        os.mkdir(os.path.join('data', 'query'))
        print("created query directory in data")
    with open(os.path.join(root_dir, 'crops.txt')) as f:
        crops = f.readlines()

    print("Download queries")
    for crop in crops:
        image_name = crop.split('/')[-1].split('.')[0]
        image_path = 'data/query/'+image_name+'.jpg'
        urllib.request.urlretrieve(crop, os.path.join(root_dir,image_path))

if __name__ == "__main__":
    main(sys.argv[1:])
