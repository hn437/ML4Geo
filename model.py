import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from definitions import TRAINING_PATH_IMG, TRAINING_PATH_MASK, TEST_PATH_IMG, TEST_PATH_MASK


# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model :)
def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model


## generator

def get_generator(batch_size, target_size):
    seed = 42
    gen_train = ImageDataGenerator(rescale=1./255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator_img = gen_train.flow_from_directory(TRAINING_PATH_IMG, batch_size=batch_size, class_mode=None,
                                                        target_size=target_size,seed=seed)
    train_generator_mask = gen_train.flow_from_directory(TRAINING_PATH_MASK, batch_size=batch_size, class_mode=None,
                                                         target_size=target_size,seed=seed)

    TRAIN_GENERATOR = zip(train_generator_img, train_generator_mask)  # combine into one to yield both at the same time


    gen_test = ImageDataGenerator(rescale=1./255.)
    test_generator_img = gen_test.flow_from_directory(TEST_PATH_IMG, batch_size=batch_size, class_mode=None,
                                                      target_size=target_size, seed=seed)
    test_generator_mask = gen_test.flow_from_directory(TEST_PATH_MASK, batch_size=batch_size, class_mode=None,
                                                       target_size=target_size, seed=seed)
    TEST_GENERATOR = zip(test_generator_img, test_generator_mask)
    return TRAIN_GENERATOR, TEST_GENERATOR


## cnn

def build_model(target_size):
    model = VGG16(input_shape=target_size + [3], include_top=False, weights="imagenet")
    # TODO: check for BATCH NORMALIZATION after each CONV Layer we can use insert_intermediate_layer_in_keras above for this purpose

    for layer in model.layers:
        layer.trainable = False  # TODO: test if a low training rate outperforms no training


    # upsampling block 6
    model.add(layers.Conv2DTranspose((2, 2), filters=512, strides=(2, 2), padding="same", name="block6_upsampling"))
    model.add(layers.Concatenate([model.get_layer("block5_pool"), model.get_layer("block6_upsampling")]))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3'))

    # upsampling block 7
    model.add(layers.Conv2DTranspose((2, 2), filters=512, strides=(2, 2), padding="same", name="block7_upsampling"))
    model.add(layers.Concatenate([model.get_layer("block4_pool"), model.get_layer("block7_upsampling")]))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv3'))

    # upsampling block 8
    model.add(layers.Conv2DTranspose((2, 2), filters=256, strides=(2, 2), padding="same", name="block8_upsampling"))
    model.add(layers.Concatenate([model.get_layer("block3_pool"), model.get_layer("block8_upsampling")]))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv2'))
