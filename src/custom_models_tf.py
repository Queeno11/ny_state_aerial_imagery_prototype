import tensorflow as tf
from tensorflow import keras
import keras_hub
from typing import Literal

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

def dinov2_model(
    resizing_size, 
    bands=3, 
    freeze_dino=True, 
    n_covariates=1, 
    head: Literal["multimodal_fusion", "late_linear_fusion", "neural_ridge", "image_only"] = "multimodal_fusion"
) -> Model:
    """
    DINOv2 (Small) model mixed with generic covariates using Keras Hub.
    Requires `pip install keras-hub`.
    """
    
    def rebuild_dinov2_top(fused):
                    
        h = layers.Dense(256, activation="gelu", name="mlp_dense_1")(fused)
        h = layers.LayerNormalization()(h)
        h = layers.Dropout(0.3)(h)
        dense64 = layers.Dense(64, activation="gelu", name="mlp_dense_2")(h)

        return dense64        
            
    # 1. IMAGE INPUT
    img_input = layers.Input(shape=(resizing_size, resizing_size, bands), name="image_input")
    x = img_input
    
    # DINOv2 strictly requires 3 bands (RGB).
    if x.shape[-1] > 3:
        print("Warning: DINOv2 expects 3 bands. Slicing to first 3 bands (RGB).")
        x = img_input[:, :, :, :3]
        
    # 2. PREPROCESSING

    # DINOv2 expects pixel values in the [0, 1] range and normalized with ImageNet statistics.
    x = layers.Rescaling(scale=1.0/255.0)(x) 

    # Normalization layer with ImageNet mean and variance (DINOv2 was trained on ImageNet, so we use those stats for normalization)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_variance = [0.229**2, 0.224**2, 0.225**2]
    x = layers.Normalization(
        mean=imagenet_mean, 
        variance=imagenet_variance,
        name="dino_normalization"
    )(x)

    # 3. FOUNDATION MODEL
    assert resizing_size == 224, "DINOv2 Small backbone requires input images of size 224x224. Please set resizing_size=224 for DINOv2 models."
    dino_base = keras_hub.models.DINOV2Backbone.from_preset(
        "dinov2_with_registers_small",
        image_shape=(224, 224, 3),
    )
    if freeze_dino:
        dino_base.trainable = False
    else:
        dino_base.trainable = True
        
    dino_outputs = dino_base({"images": x})
    
    if isinstance(dino_outputs, dict):
        cls_token = dino_outputs.get("class_token", dino_outputs.get("sequence_output")[:, 0, :])
    else:
        cls_token = dino_outputs[:, 0, :]

    if head=="image_only" and n_covariates > 0:
        raise ValueError("Image-only head cannot have covariates. Please set n_covariates=0 for head='image_only'.")
    
    # ==========================================
    # HEAD 0: Image only
    # ==========================================
    if head == "image_only":
        fused = cls_token
        dense64 = rebuild_dinov2_top(fused)
        output = layers.Dense(1, activation="linear", name="predictions")(dense64)
        model = Model(inputs=img_input, outputs=output, name="Dinov2_ImageOnly")
        return model
        
    # ==========================================
    # HEAD 1: CS Standard (Multimodal Fusion)
    # ==========================================
    elif head == "multimodal_fusion":
        commute_input = layers.Input(shape=(n_covariates,), name="covariates")
        fused = layers.Concatenate(name="concat_features")([cls_token, commute_input])
        dense64 = rebuild_dinov2_top(fused)
        output = layers.Dense(1, activation="linear", name="predictions")(dense64)
        model = Model(inputs=[img_input, commute_input], outputs=output, name="Dinov2_Multimodal")

    # ==========================================
    # HEAD 2: Mixed Approach (Late Linear Fusion)
    # ==========================================
    elif head == "late_linear_fusion":
        visual_wealth_feature = rebuild_dinov2_top(cls_token)  # This is the image-only regression output from the DINOv2 features

        commute_input = layers.Input(shape=(n_covariates,), name="covariates")
        fused = layers.Concatenate(name="concat_final")([visual_wealth_feature, commute_input])
        output = layers.Dense(1, activation="linear", name="linear_regression_predictions")(fused) # Linear regression (with activation) between the visual wealth feature and the covariates
        
        model = Model(inputs=[img_input, commute_input], outputs=output, name="Dinov2_LateLinear")

    # ==========================================
    # HEAD 3: Pure Econ (Neural Ridge Regression)
    # ==========================================
    elif head == "neural_ridge":
        # Pure linear projection of the 384 DINOv2 dimensions using an L2 penalty (Ridge equivalent)
        visual_wealth_feature = layers.Dense(
            1, 
            activation="linear", 
            kernel_regularizer=keras.regularizers.l2(0.01),
            name="visual_ridge_projection"
        )(cls_token)

        commute_input = layers.Input(shape=(n_covariates,), name="covariates")
        fused = layers.Concatenate(name="concat_final")([visual_wealth_feature, commute_input])
        output = layers.Dense(1, activation="linear", name="predictions")(fused)
        
        model = Model(inputs=[img_input, commute_input], outputs=output, name="Dinov2_NeuralRidge")


    else:
        raise ValueError(f"Unknown head type: {head}")

    return model

def build_siamese_dinov2(
    resizing_size, 
    n_covariates, 
    head_type="multimodal_fusion"
) -> Model:
    """
    Wraps the dinov2_model into a Siamese architecture for pairwise comparison.
    """
    
    # 1. Instantiate the base model ONCE. 
    # This is the template for both towers and ensures weight sharing.
    # The head (g_phi) and the frozen backbone (f_theta) are contained within.
    base_model = dinov2_model(
        resizing_size=resizing_size,
        n_covariates=n_covariates,
        head=head_type,
        freeze_dino=True 
    )

    # 2. Define the inputs for the two towers (A and B)
    # Tower A Inputs
    img_A = layers.Input(shape=(resizing_size, resizing_size, 3), name="image_A")
    
    # Tower B Inputs
    img_B = layers.Input(shape=(resizing_size, resizing_size, 3), name="image_B")

    # Handle covariates if they exist
    if n_covariates > 0:
        cov_A = layers.Input(shape=(n_covariates,), name="covariates_A")
        cov_B = layers.Input(shape=(n_covariates,), name="covariates_B")
        
        # Pass inputs through the shared base_model
        score_A = base_model([img_A, cov_A])
        score_B = base_model([img_B, cov_B])
        
        # Define the full model's inputs
        model_inputs = [img_A, cov_A, img_B, cov_B]
        
    else: # Image-only case
        # Pass inputs through the shared base_model
        score_A = base_model(img_A)
        score_B = base_model(img_B)
        
        # Define the full model's inputs
        model_inputs = [img_A, img_B]

    # 3. Define the final output for the Margin Ranking Loss
    # The loss function operates on the difference between the two ranking scores.
    # This layer simply calculates that difference for the model's output.
    # Note: The actual loss calculation happens outside the model, during compilation/training.
    output_diff = layers.Subtract(name="ranking_score_difference")([score_A, score_B])

    # 4. Create the final Siamese Model
    siamese_network = Model(
        inputs=model_inputs,
        outputs=output_diff,
        name=f"Siamese_{head_type}_Dinov2"
    )
    
    return siamese_network

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
