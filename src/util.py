import numpy as np
import keras.preprocessing.image as image
from keras.applications.vgg16 import preprocess_input

def normalize(v):
    """L2 normalization of vector"""
    norm = np.linalg.norm(v, 2)
    if norm == 0.0:
        return v
    else:
        return v / norm


def loadImage(image_path):
    """
    Reads image path into PIL image and converts into input format for the model

    Returns: returns an image as numpy array.

    """

    img = image.load_img(image_path)
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)



    return img
