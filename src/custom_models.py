import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    MobileNetV2,
    MobileNetV3Small,
    Xception,
    MobileNetV3Large,
    EfficientNetB0,
    EfficientNetV2B1,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
    ResNet152V2,
)


def rebuild_top(model_base, kind="cla", legacy=False) -> Sequential:
    """Rebuild top of a pre-trained model to make it suitable for classification or regression."""

    assert kind in ["cla", "reg"], "kind must be either cla or reg"

    model = tf.keras.Sequential()

    model.add(model_base)

    if legacy is False:
        # Rebuild top
        #   Based on: https://github.com/MarkusRosen/keras-efficientnet-regression/blob/master/efficient_net_keras_regression.py    
        model.add(layers.GlobalAveragePooling2D(name="avg_pool"))
        model.add(layers.BatchNormalization())
        top_dropout_rate = 0.4
        model.add(layers.Dropout(top_dropout_rate, name="top_dropout"))

    else: 
        model.add(layers.Flatten())

    if kind == "cla":
        # Add fully conected layers
        # model.add(layers.Dense(2048, name="fc1", activation="relu"))
        #         model.add(layers.Dense(2048, name="fc1", activation="relu"))
        model.add(layers.Dense(10, name="predictions", activation="softmax"))
    if kind == "reg":
        model.add(layers.Dense(1, name="predictions", activation="linear"))

    base = model.layers[0]            # base model is the first layer in your Sequential
    base.trainable = True

    return model


def mobnet_v3_large(resizing_size, bands=8, kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/mobilenet_v3/#mobilenetv3small-function"""

    model_base = MobileNetV3Large(
        include_top=False,
        input_shape=(resizing_size, resizing_size, bands),
        weights=None,
        include_preprocessing=False,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def efficientnet_v2S(resizing_size, bands=8, kind="reg", weights=None) -> Sequential:

    model_base = EfficientNetV2S(
        include_top=False,
        input_shape=(resizing_size, resizing_size, bands),
        weights=weights,
        include_preprocessing=False,
    )

    model = rebuild_top(model_base, kind=kind, legacy=True)
    return model


def efficientnet_v2M(resizing_size, bands=8, kind="reg", weights=None) -> Sequential:

    model_base = EfficientNetV2M(
        include_top=False,
        input_shape=(resizing_size, resizing_size, bands),
        weights=weights,
        include_preprocessing=False,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def efficientnet_v2B1(resizing_size, bands=8, kind="reg", weights=None) -> Sequential:

    model_base = EfficientNetV2B1(
        include_top=False,
        input_shape=(resizing_size, resizing_size, bands),
        weights=weights,
        include_preprocessing=False,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def small_cnn(resizing_size=200) -> Sequential:
    """layer normalization entre cada capa y su activación. Batch norm no funca
    porque uso batches de 1, se supone que no funciona bien para muestras de
    menos de 32 (demasiada varianza en las estadísticas de cada batch).

    'there are strong theoretical reasons against it, and multiple publications
    have shown BN performance degrade for batch_size under 32, and severely for <=8.
    In a nutshell, batch statistics "averaged" over a single sample vary greatly
    sample-to-sample (high variance), and BN mechanisms don't work as intended'
    (https://stackoverflow.com/questions/59648509/batch-normalization-when-batch-size-1).

    Layer normalization is independent of the batch size, so it can be applied to
    batches with smaller sizes as well.
    (https://www.pinecone.io/learn/batch-layer-normalization/)"""

    model = models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="linear",
            input_shape=(resizing_size, resizing_size, 4),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    # model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))

    return model


def spatialecon_cnn(resizing_size, bands=8):
    def conv_block(inputs, n_filter, regularizer, common_args):
        x = tf.keras.layers.Conv2D(
            filters=n_filter,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
            **common_args
        )(inputs)
        x = tf.keras.layers.Conv2D(
            filters=n_filter,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
            **common_args
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=n_filter,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
            **common_args
        )(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        return x

    def dense_block(inputs, n_filter, regularizer, drop_rate, common_args):
        x = tf.keras.layers.Dense(
            16 * n_filter,
            activation="relu",
            kernel_regularizer=regularizer,
            **common_args
        )(inputs)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        x = tf.keras.layers.Dense(
            16 * n_filter,
            activation="relu",
            kernel_regularizer=regularizer,
            **common_args
        )(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        x = tf.keras.layers.Dense(
            8 * n_filter,
            activation="relu",
            kernel_regularizer=regularizer,
            **common_args
        )(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        return x

    def make_level_model(img_size, n_bands, nf, dr):
        regularizer = tf.keras.regularizers.l2(0.0001)
        initializer = tf.keras.initializers.glorot_normal()
        common_args = {"kernel_initializer": initializer}

        inputs = tf.keras.layers.Input(shape=(img_size, img_size, n_bands))
        x = conv_block(inputs, nf, regularizer, common_args)
        x = conv_block(x, nf * 2, regularizer, common_args)
        x = conv_block(x, nf * 4, regularizer, common_args)

        x = tf.keras.layers.Flatten()(x)
        x = dense_block(x, nf, regularizer, dr, common_args)
        output = tf.keras.layers.Dense(1, **common_args)(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    return make_level_model(
        resizing_size, bands, 32, 0.5
    )  # 32 filters, 0.5 dropout rate	- original parameters from the paper


def unfreeze_base_model(model: tf.keras.Model) -> bool:
    """Unfreeze the nested base (functional) model if present, or the top-level layers.

    This sets `.trainable = True` on either the first nested `tf.keras.Model` found
    or on all top-level layers. Returns True if operation completed, False on exception.
    """
    try:
        base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
        if base is None:
            model.trainable = True
            for lay in model.layers:
                lay.trainable = True
        else:
            base.trainable = True
            for lay in base.layers:
                lay.trainable = True
        return True
    except Exception:
        try:
            model.trainable = True
        except Exception:
            pass
        return False
