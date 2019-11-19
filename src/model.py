
def loadModel():
    from keras.applications.vgg16 import VGG16
    from keras.models import Model

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(None,None,3))
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model
