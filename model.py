import tensorflow as tf
from tensorflow.keras import Model, Sequential, activations, layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from definitions import TEST_PATH, TRAINING_PATH, TRAINING_PATH_IMG


## generator
def get_generator(batch_size, target_size):
    seed = 42
    gen_train_img = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    gen_train_mask = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_generator_img = gen_train_img.flow_from_directory(
        TRAINING_PATH,
        classes=["img"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=True,
    )
    train_generator_mask = gen_train_mask.flow_from_directory(
        TRAINING_PATH,
        classes=["mask"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=True,
    )
    no_of_trainsets = train_generator_img.samples
    TRAIN_GENERATOR = zip(
        train_generator_img, train_generator_mask
    )  # combine into one to yield both at the same time

    gen_test_img = ImageDataGenerator(rescale=1.0 / 255.0)
    gen_test_mask = ImageDataGenerator()
    test_generator_img = gen_test_img.flow_from_directory(
        TEST_PATH,
        classes=["img"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=False,
    )
    test_generator_mask = gen_test_mask.flow_from_directory(
        TEST_PATH,
        classes=["mask"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=False,
    )
    no_of_validsets = test_generator_img.samples
    TEST_GENERATOR = zip(test_generator_img, test_generator_mask)
    return TRAIN_GENERATOR, TEST_GENERATOR, no_of_trainsets, no_of_validsets


## cnn
def conv_block(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def decoder_block(input, skip_features, num_filters, no_of_conv_blocks):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = layers.Concatenate()([x, skip_features])
    for _ in range(no_of_conv_blocks):
        x = conv_block(x, num_filters)
    return x


def build_model(target_size):
    inputs = layers.Input(shape=target_size + [3])

    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    # make the first pretrained layer untrainable
    for layer in vgg16.layers[:-8]:
        layer.trainable = (
            False  # TODO: test if a low training rate outperforms no training
        )

    # encoder layer
    e1 = vgg16.get_layer("block1_conv2").output
    e2 = vgg16.get_layer("block2_conv2").output
    e3 = vgg16.get_layer("block3_conv3").output
    e4 = vgg16.get_layer("block4_conv3").output
    e5 = vgg16.get_layer("block5_conv3").output

    # bottom layer
    last_pool = vgg16.get_layer("block5_pool").output

    # decoder layer
    d1 = decoder_block(last_pool, e5, 512, 3)
    d2 = decoder_block(d1, e4, 512, 3)
    d3 = decoder_block(d2, e3, 256, 3)
    d4 = decoder_block(d3, e2, 128, 2)
    d5 = decoder_block(d4, e1, 64, 2)

    # output
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d5)

    # initiating model
    model = Model(inputs, outputs, name="ML4Geo")

    # compile model
    model.compile(loss=BinaryCrossentropy(), optimizer=Nadam(), metrics=["accuracy"])

    return model
