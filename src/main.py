##############      Configuración      ##############
import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src import grid_predictions
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, IMAGERY_ROOT
pd.set_option("display.max_columns", None)



# path_programas  = globales[7]
###############################################

import src.true_metrics as true_metrics
import src.custom_models as custom_models
import src.build_dataset as build_dataset
import src.grid_predictions as  grid_predictions
import src.geo_utils as geo_utils

import os
import sys
import json
import scipy
import random
import pandas as pd
import xarray as xr
import warnings
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from keras import backend as K

mixed_precision.set_global_policy('mixed_float16')

# Mute TF low_level warnings: https://stackoverflow.com/questions/76912213/tf2-13-local-rendezvous-recv-item-cancelled
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)
from tensorflow.keras.models import Sequential
# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")

# Define a subset of the data that will comfortably fit in RAM cache
CACHE_SIZE = 800*128 # Around 100k images (128 batch size)

# Disable
def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def generate_savename(
    model_name, image_size, learning_rate, stacked_images, years, extra
):
    years_str = "-".join(map(str, years))
    if len(stacked_images) > 1:
        stacked_str = "-".join(map(str, stacked_images))
        savename = f"{model_name}_lr{learning_rate}_size{image_size}_y{years_str}_stack{stacked_str}{extra}"
    else:
        savename = (
            f"{model_name}_lr{learning_rate}_size{image_size}_y{years_str}{extra}"
        )

    return savename


def open_datasets(sat_data="aerial", years=[2013, 2018, 2022]):

    ### Open dataframe with files and labels
    print("Reading dataset...")
    df = build_dataset.load_income_dataset()

    year_cols = []
    datasets_all_years = {}
    extents_all_years = {}
    for year in years:
        if sat_data == "aerial":
            sat_imgs_datasets, extents = build_dataset.load_satellite_datasets(
                year=year
            )
        elif sat_data == "landsat":
            raise NotImplementedError("Landsat support not implemented yet.")
            sat_imgs_datasets, extents = build_dataset.load_landsat_datasets()

        df = build_dataset.assign_datasets_to_gdf(df, extents, year=year, verbose=True)
        datasets_all_years[year] = sat_imgs_datasets
        extents_all_years[year] = extents
        year_cols += [f"dataset_{year}"]

    df = df[df[year_cols].notna().any(axis=1)]
    print("Datasets loaded!")

    return datasets_all_years, extents_all_years, df


def create_train_test_dataframes(df, savename, small_sample=False):
    """Create train and test dataframes with the links and xr.datasets to use for training and testing

    Load the ICPAG dataset and assign the links to the corresponding xr.dataset, then split the census tracts
    into train and test. The train and test dataframes contain the links and xr.datasets to use for training and
    testing.
    """
    if small_sample:
        df = df.sample(1000, random_state=825).reset_index(drop=True)

    ### Split census tracts based on train/test
    #       (the hole census tract must be in the corresponding region)
    df = build_dataset.split_train_test(df)
    df = df[
        ["GEOID", "var", "type", "geometry"]
        + [col for col in df.columns if "dataset" in col]
    ]

    ### Train/Test
    list_of_datasets = []

    df_test = df[df["type"] == "test"].copy().reset_index(drop=True)
    assert df_test.shape[0] > 0, f"Empty test dataset!"
    df_train = df[df["type"] == "train"].copy().reset_index(drop=True)
    assert df_train.shape[0] > 0, f"Empty train dataset!"

    test_dataframe_path = PROCESSED_DATA_DIR / "test_datasets" / f"{savename}_test_dataframe.feather"
    df_test.to_feather(test_dataframe_path)
    print("Se creó el archivo:", test_dataframe_path)

    train_dataframe_path = PROCESSED_DATA_DIR / "train_datasets" / f"{savename}_train_dataframe.feather"
    df_train.to_feather(train_dataframe_path)
    print("Se creó el archivo:", train_dataframe_path)

    return df_train, df_test


def create_datasets(
    df_not_test,
    df_test,
    all_years_datasets,
    image_size,
    resizing_size,
    sample=1,
    nbands=4,
    tiles=1,
    stacked_images=[1],
    savename="",
    save_examples=True,
):
    available_years = list(all_years_datasets.keys())
    
    # --- Configuration ---
    DEBUG_DIR = PROCESSED_DATA_DIR / "debug_examples"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    fname = f"{savename}_example"

    # EPOCHS_PER_CYCLE: How many epochs to reuse the cache before generating new crops
    # 1 Slow Epoch (Gen) + 9 Fast Epochs (Read) = 10 Total  
    READ_BATCH_SIZE = 16
    TRAIN_BATCH_SIZE = 128
    
    # Shapes
    OUTPUT_SHAPE_IMG = (None, resizing_size, resizing_size, nbands * len(stacked_images))
    OUTPUT_SHAPE_LBL = (None,)

    # --- 1. The Python Data Loader (Slow Logic) ---
    def get_mini_batch_data(batch_indices, df_subset):
        indices = batch_indices.numpy()
        batch_imgs = []
        batch_lbls = []
        
        # Randomly pick a year for this batch to optimize Zarr access
        batch_year = random.choice(available_years)
        primary_dataset = all_years_datasets[batch_year]
        
        total_bands = nbands * len(stacked_images)
        target_shape = (total_bands, image_size, image_size)

        for i in indices:
            # ... (Standard extraction logic) ...
            try:
                polygon = df_subset.loc[i]["geometry"]
                value = df_subset.loc[i]["var"]
                
                # Logic to get dataset
                dataset_name = df_subset.loc[i][f"dataset_{batch_year}"]
                if not pd.isna(dataset_name):
                    link_dataset = primary_dataset[dataset_name]
                else:
                    link_dataset = None # (Fallback logic omitted for brevity, add back if needed)

                image = np.zeros(shape=(nbands, 0, 0))
                if link_dataset is not None:
                    # RANDOM CROP HAPPENS HERE
                    image, _ = geo_utils.stacked_image_from_census_tract(
                        dataset=link_dataset,
                        polygon=polygon,
                        img_size=image_size,
                        n_bands=nbands,
                        stacked_images=stacked_images,
                    )
                
                if image.shape != target_shape:
                    #  print("Image shape mismatch:", image.shape, "expected:", target_shape)
                     image = np.zeros(shape=(resizing_size, resizing_size, total_bands))

                else:
                     image = geo_utils.process_image(image, resizing_size)
                     # NO AUGMENTATION HERE (We cache the clean image)

                batch_imgs.append(image)
                batch_lbls.append(value)
                
            except Exception as e:
                print(e)
                # Fail-safe
                batch_imgs.append(np.zeros((resizing_size, resizing_size, total_bands)))
                batch_lbls.append(0.0)

        if save_examples and not os.path.exists(DEBUG_DIR / f"{fname}_img.npy"):
            np.save(DEBUG_DIR / f"{fname}_img.npy", np.stack(batch_imgs).astype(np.uint8))
            np.save(DEBUG_DIR / f"{fname}_lbl.npy", np.stack(batch_lbls).astype(np.float32))

        return np.stack(batch_imgs).astype(np.uint8), np.stack(batch_lbls).astype(np.float32)

    # --- 2. GPU Augmentation ---
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomContrast(0.2),
    ])

    def apply_augmentation(img, lbl):
        return data_augmentation(img, training=True), lbl

    # --- 3. Cache Management Helper ---
    def clean_cache_file(filepath):
        """Deletes cache files from previous runs to free space."""
        # We need this to run as a tf.py_function inside the graph
        path_str = filepath.numpy().decode('utf-8')
        if os.path.exists(path_str + ".index"):
            try:
                os.remove(path_str + ".index")
                os.remove(path_str + ".data-00000-of-00001")
            except: pass
        return np.int32(0) # Dummy return

    # --- 4. The Cyclic Pipeline Builder ---
    def build_cyclic_dataset(df_subset, subset_name, is_train=False):
        
        # For Validation/Test, we don't need rotation. Just cache once.
        if not is_train:
            ds = tf.data.Dataset.from_tensor_slices(list(range(df_subset.shape[0])))
            ds = ds.batch(READ_BATCH_SIZE)
            ds = ds.map(lambda i: tf.py_function(lambda x: get_mini_batch_data(x, df_subset), [i], [tf.uint8, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
            def set_sh(img, lbl):
                img.set_shape(OUTPUT_SHAPE_IMG); lbl.set_shape(OUTPUT_SHAPE_LBL); return img, lbl
            ds = ds.map(set_sh).unbatch()
            
            cache_file = str(CACHE_DIR / f"{savename}_{subset_name}_static.tfcache")
            # We must wrap the python call to clean the cache in a py_function
            _ = tf.py_function(clean_cache_file, [cache_file], tf.int32)
            ds = ds.cache(cache_file)
            ds = ds.batch(TRAIN_BATCH_SIZE)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        # --- TRAIN LOGIC (GRAPH-COMPATIBLE ROTATION) ---
        
        def make_one_cycle(cycle_index):
            # Use tf.print for graph-compatible logging
            tf.print("Starting data generation for cycle:", cycle_index)

            # 1. GRAPH-COMPATIBLE SAMPLING:
            # Instead of df.sample(), we shuffle the full index list and take a subset.
            all_indices_ds = tf.data.Dataset.from_tensor_slices(df_subset.index.to_numpy())
            
            # Shuffle the full list of indices. The seed ensures we get a DIFFERENT shuffle for each cycle.
            sampled_indices_ds = all_indices_ds.shuffle(
                len(df_subset), 
                seed=tf.cast(cycle_index, tf.int64), # Use cycle_index as the seed
                reshuffle_each_iteration=False # We want one consistent sample per cycle
            ).take(CACHE_SIZE)
            
            # 2. Determine Cache Filename (Toggle between A and B)
            cycle_id = tf.cast(cycle_index % 2, tf.int32)
            cache_filename = tf.strings.join([str(CACHE_DIR), f"/{savename}_train_cycle_", tf.strings.as_string(cycle_id), ".tfcache"])
            
            # 3. Clear previous cache
            _ = tf.py_function(clean_cache_file, [cache_filename], tf.int32)
            
            # 4. Map the loader function onto our sampled indices
            # The loader needs the original dataframe, which we can access via closure
            ds = sampled_indices_ds.batch(READ_BATCH_SIZE)
            ds = ds.map(
                lambda indices: tf.py_function(
                    # We use the full df_subset and tell the loader which indices to use
                    func=lambda x: get_mini_batch_data(x, df_subset.loc[x.numpy()]),
                    inp=[indices],
                    Tout=[tf.uint8, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            def set_shapes(imgs, lbls):
                imgs.set_shape(OUTPUT_SHAPE_IMG); lbls.set_shape(OUTPUT_SHAPE_LBL); return imgs, lbls
            ds = ds.map(set_shapes).unbatch()
            
            # 5. CACHE THE SUBSET
            ds = ds.cache(tf.strings.as_string(cache_filename)) # Cast to string tensor
            
            # 6. Repeat, Shuffle, Batch, Augment
            ds = ds.shuffle(1000) 
            ds = ds.batch(TRAIN_BATCH_SIZE)
            ds = ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            CYCLES_TO_SURVIVE = 5 
            ds = ds.repeat(CYCLES_TO_SURVIVE)            
            return ds

        # Create the infinite stream of 1-epoch cycles
        num_epochs = 150 # Your total epochs
        master_ds = tf.data.Dataset.range(num_epochs).flat_map(make_one_cycle)
        master_ds = master_ds.prefetch(tf.data.AUTOTUNE)
        return master_ds
        
    # --- Execution ---
    
    # Shuffle everything so images are not sequential:
    df_not_test = df_not_test.sample(frac=1.0, random_state=825).reset_index(drop=True)
    df_test = df_test.sample(frac=1.0, random_state=825).reset_index(drop=True)
    
    # Generate validation dataset
    df_val = df_not_test.sample(frac=0.066667, random_state=200)
    df_train = df_not_test.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    
    print("\nSetting up Auto-Rotating Cached Pipelines...")

    if sample > 1:
        # For cache logic, simpler to concat dataframe than repeat dataset
        df_train = pd.concat([df_train]*sample).reset_index(drop=True)

    train_dataset = build_cyclic_dataset(df_train, "train", is_train=True)
    val_dataset = build_cyclic_dataset(df_val, "val", is_train=False)
    test_dataset = build_cyclic_dataset(df_test, "test", is_train=False)

    print("Datasets Ready!")
    return train_dataset, val_dataset, test_dataset

def get_callbacks(
    savename,
    logdir=None,
) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """

    class CustomLossCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_dir, savename):
            super(CustomLossCallback, self).__init__()
            self.log_dir = log_dir
            self.savename = savename
            self.param_log_path = f"{MODELS_DIR}/models_by_epoch/{savename}/{savename}_optimizer_params.csv"

        def on_epoch_end(self, epoch, logs=None):
            # 1. Save model
            epoch_dir = f"{MODELS_DIR}/models_by_epoch/{self.savename}"
            os.makedirs(epoch_dir, exist_ok=True)
            model_path = f"{epoch_dir}/{self.savename}_{epoch}.keras"
            self.model.save(model_path, include_optimizer=True)

            # 2. Extract Optimizer Parameters
            opt = self.model.optimizer
            
            # In TF/Keras, some attributes might be tracked as variables or simple floats
            # This handles both cases safely
            def get_val(attr):
                val = getattr(opt, attr, "N/A")
                if hasattr(val, "numpy"):
                    return val.numpy()
                return val

            current_params = {
                "epoch": epoch,
                "learning_rate": get_val("learning_rate"),
                "beta_1": get_val("beta_1"),
                "beta_2": get_val("beta_2"),
                "epsilon": get_val("epsilon"),
                "iterations": get_val("iterations")
            }

            # 3. Store to CSV
            df_params = pd.DataFrame([current_params])
            if not os.path.isfile(self.param_log_path):
                df_params.to_csv(self.param_log_path, index=False)
            else:
                df_params.to_csv(self.param_log_path, mode='a', header=False, index=False)

    tensorboard_callback = TensorBoard(
        log_dir=logdir, histogram_freq=1, profile_batch=0
    )
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    # Create an instance of your custom callback
    custom_loss_callback = CustomLossCallback(log_dir=logdir, savename=savename)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,  # the training is terminated as soon as the performance measure gets worse from one epoch to the next
        start_from_epoch=50,
        patience=50,  # amount of epochs with no improvements until the model stops
        verbose=2,
        mode="auto",  # the model is stopped when the quantity monitored has stopped decreasing
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=10, min_lr=0.000001
    )
    model_checkpoint_callback = ModelCheckpoint(
        f"{MODELS_DIR}/{savename}.keras",
        monitor="loss",
        verbose=1,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )
    csv_logger = CSVLogger(
        f"{MODELS_DIR}/models_by_epoch/{savename}/{savename}_history.csv", append=True
    )

    return [
        tensorboard_callback,
        # reduce_lr,
        # early_stopping_callback,
        model_checkpoint_callback,
        csv_logger,
        custom_loss_callback,
    ]


def train_model(
    model_function: Model,
    lr: float,
    train_dataset: Iterator,
    val_dataset: Iterator,
    loss: str,
    epochs: int,
    metrics: List[str],
    callbacks: List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]],
    savename: str = "",
    logdir: str = "",
):
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_function : Model
        Keras model function like small_cnn()  or adapt_efficient_net().
    lr : float
        Learning rate.
    train_dataset : Iterator
        tensorflow dataset for the training data.
    test_dataset : Iterator
        tesorflow dataset for the test data.
    loss: str
        Loss function.
    metrics: List[str]
        List of metrics to be used.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history.
    """
    import tensorflow.python.keras.backend as K

    def get_last_trained_epoch(savename):

        model_dir = MODELS_DIR / "models_by_epoch" / f"{savename}"
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            epochs = [file.split("_")[-1] for file in files]
            epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
            if not epochs:
                # Directory exists but has no saved model epochs (only history.csv or similar), remove it to start fresh
                shutil.rmtree(model_dir)
                os.makedirs(model_dir)
                print("Model not found, running from begining")
                initial_epoch = None

        else:
            os.makedirs(model_dir)
            print("Model not found, running from begining")
            initial_epoch = None

        return initial_epoch

    initial_epoch = get_last_trained_epoch(savename)

    if initial_epoch is None:
        # constructs the model and compiles it
        model = model_function
        model.summary()
        # keras.utils.plot_model(model, to_file=model_name + ".png", show_shapes=True)

        # optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)
        initial_epoch = 0

    else:
        print("Restoring model...")
        try:
            model_path = (
                f"{MODELS_DIR}/models_by_epoch/{savename}/{savename}_{initial_epoch}"
            )
            model = keras.models.load_model(model_path)  # load the model from file
        except:
            initial_epoch -= 1
            model_path = (
                f"{MODELS_DIR}/models_by_epoch/{savename}/{savename}_{initial_epoch}"
            )
            model = keras.models.load_model(model_path)  # load the model from file
        initial_epoch = initial_epoch + 1

    # The number of steps is the number of samples divided by batch size
    validation_steps = 100 # Use your TRAIN_BATCH_SIZE
    steps_per_epoch = CACHE_SIZE // 128

    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch, # Total train buildings in NYC = 328875 x 
        initial_epoch=initial_epoch,
        # validation_data=val_dataset,
        # validation_steps=validation_steps,
        callbacks=callbacks,
    )

    return model, history  # type: ignore


def plot_predictions_vs_real(df):
    """Genera un scatterplot con la comparación entre los valores reales y los predichos.

    Parameters:
    -----------
    - df: DataFrame con las columnas 'real' y 'pred'. Se recomienda utilizar el test Dataset para validar
    la performance del modelo.

    """

    # Resultado general
    slope, intercept, r, p_value, std_err = scipy.stats.linregress(
        df["real"], df["pred"]
    )

    # Gráfico de correlacion
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    g = sns.jointplot(
        data=df,
        x="real",
        y="pred",
        kind="reg",
        xlim=(df.real.min(), df.real.max()),
        ylim=(df.real.min(), df.real.max()),
        height=10,
        joint_kws={"line_kws": {"color": "cyan"}},
        scatter_kws={"s": 2},
    )
    g.ax_joint.set_xlabel("Valores Reales", fontweight="bold")
    g.ax_joint.set_ylabel("Valores Predichos", fontweight="bold")

    # Diagonal
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, "-r", color="orange", linewidth=2)

    # Texto con la regresión
    plt.text(
        0.1,
        0.9,
        f"$y={intercept:.2f}+{slope:.2f}x$; $R^2$={r**2:.2f}",
        transform=g.ax_joint.transAxes,
        fontsize=12,
    )

    return g


def validate_parameters(params, default_params):

    for key, value in params.items():
        if key not in default_params.keys():
            raise ValueError("Invalid parameter: %s" % key)

    try:
        sat_data = params["sat_data"]
    except:
        print("No parameters are being validated, as sat_data is not defined...")
        return

    nbands = params["nbands"]
    years = params["years"]
    resizing_size = params["resizing_size"]

    sat_options = ["aerial", "pleiades", "landsat"]
    if sat_data not in sat_options:
        raise ValueError("Invalid sat_data type. Expected one of: %s" % sat_options)

    if sat_data == "pleiades":
        if (nbands != 3) and (nbands != 4):
            raise ValueError("nbands for pleiades dataset must be 3 or 4.")

        # if len(years) > 3:
        #     raise ValueError("Pleiades data only available in 2013, 2018 and 2022.")
        # elif not all(year in [2013, 2018, 2022] for year in years):
        #     raise ValueError("Pleiades data only available in 2013, 2018 and 2022.")

        if resizing_size > 1024:
            warnings.warn(
                "Warning: resizing_size greater than 1024 might encompass an area much bigger than the census tracts..."
            )

    elif sat_data == "landsat":
        if nbands > 10:
            raise ValueError("nbands for pleiades dataset must be less than 11.")

        if years != [2013]:
            raise ValueError("Landsat data only available in 2013.")

        if resizing_size > 32:
            warnings.warn(
                "Warning: resizing_size greater than 32 might encompass an area much bigger than the census tracts..."
            )

    return


def fill_params_defaults(params):

    default_params = {
        "model_name": "effnet_v2S",
        "kind": "reg",
        "weights": None,
        "image_size": 256,
        "resizing_size": 128,
        "tiles": 1,
        "nbands": 4,
        "stacked_images": [1],
        "sample_size": 5,
        "small_sample": False,
        "n_epochs": 100,
        "learning_rate": 0.0001,
        "sat_data": "pleiades",
        "years": [2013],
        "extra": "",
    }
    validate_parameters(params, default_params)

    # Merge default and provided hyperparameters (keep from params)
    updated_params = {**default_params, **params}
    print(updated_params)

    return updated_params


def set_model_and_loss_function(
    model_name: str, kind: str, resizing_size: int, weights: str, bands: int = 4
):
    # Diccionario de modelos
    get_model_from_name = {
        "small_cnn": custom_models.small_cnn(resizing_size),  # kind=kind),
        "mobnet_v3_large": custom_models.mobnet_v3_large(
            resizing_size, bands=bands, kind=kind, weights=weights
        ),
        "effnet_v2S": custom_models.efficientnet_v2S(
            resizing_size, bands=bands, kind=kind, weights=weights
        ),
        "effnet_v2M": custom_models.efficientnet_v2M(
            resizing_size, bands=bands, kind=kind, weights=weights
        ),
        "effnet_v2B1": custom_models.efficientnet_v2B1(
            resizing_size, bands=bands, kind=kind, weights=weights
        ),
        "spatialecon_cnn": custom_models.spatialecon_cnn(
            resizing_size,
            bands=bands,
        ),
    }

    # Validación de parámetros
    assert kind in ["reg", "cla"], "kind must be either 'reg' or 'cla'"
    # assert (
    #     model_name in get_model_from_name.keys()
    # ), "model_name must be one of the following: " + str(
    #     list(get_model_from_name.keys())
    # )

    # Get model
    model = get_model_from_name[model_name]
        # Set loss and metrics
    if kind == "reg":
        loss = keras.losses.MeanSquaredError()
        # loss = keras.losses.MeanAbsoluteError()
        metrics = [
            # keras.metrics.MeanAbsoluteError(),
            keras.metrics.R2Score(),
            # keras.metrics.RootMeanSquaredError(),
            # keras.metrics.MeanAbsolutePercentageError(),
        ]

    elif kind == "cla":
        loss = keras.losses.CategoricalCrossentropy()
        metrics = [
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.CategoricalCrossentropy(),
        ]

    return model, loss, metrics


def generate_parameters_log(params, savename):

    os.makedirs(f"{MODELS_DIR}/{savename}", exist_ok=True)
    filename = f"{MODELS_DIR}/{savename}/{savename}_logs.txt"

    with open(filename, "w") as file:
        json.dump(params, file)

    print(f"Se creó {filename} con los parametros utilizados.")
    return


def run(
    params=None,
    train=True,
    compute_loss=True,
    generate_grid=False,
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    params = fill_params_defaults(params)

    model_name = params["model_name"]
    kind = params["kind"]
    weights = params["weights"]
    image_size = params["image_size"]
    resizing_size = params["resizing_size"]
    tiles = params["tiles"]
    nbands = params["nbands"]
    stacked_images = params["stacked_images"]
    sample_size = params["sample_size"]
    small_sample = params["small_sample"]
    n_epochs = params["n_epochs"]
    learning_rate = params["learning_rate"]
    sat_data = params["sat_data"]
    years = params["years"]
    extra = params["extra"]

    #
    savename = generate_savename(
        model_name, image_size, learning_rate, stacked_images, years, extra
    )
    log_dir = f"{LOGS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    generate_parameters_log(params, savename)

    all_years_datasets, all_years_extents, df = open_datasets(
        sat_data=sat_data, years=[2022] # FIXME! 
    )

    if train:

        ## Set Model & loss function
        model, loss, metrics = set_model_and_loss_function(
            model_name=model_name,
            kind=kind,
            bands=nbands * len(stacked_images),
            resizing_size=resizing_size,
            weights=weights,
        )

        # ### Create train and test dataframes from ICPAG
        df_not_test, df_test = create_train_test_dataframes(
            df, savename, small_sample=small_sample
        )

        ## Transform dataframes into datagenerators:
        #    instead of iterating over census tracts (dataframes), we will generate one (or more) images per census tract
        print("Setting up data generators...")
        train_dataset, val_dataset, test_dataset = create_datasets(
            df_not_test=df_not_test,
            df_test=df_test,
            all_years_datasets=all_years_datasets,
            image_size=image_size,
            resizing_size=resizing_size,
            nbands=nbands,
            stacked_images=stacked_images,
            tiles=tiles,
            sample=sample_size,
            savename=savename,
            save_examples=True,
        )
        # Get tensorboard callbacks and set the custom test loss computation
        #   at the end of each epoch
        callbacks = get_callbacks(
            savename=savename,
            logdir=log_dir,
        )

        # Run model
        model, history = train_model(
            model_function=model,
            lr=learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss=loss,
            metrics=metrics,
            callbacks=callbacks,
            epochs=n_epochs,
            savename=savename,
            logdir=log_dir,
        )
        print("Fin del entrenamiento")

    ## Compute metrics
    # Genero la test_loss por RC
    if compute_loss:
        true_metrics.compute_loss(  # No entra el test_dataset acá pero despues usa el df_test guardado en memoria
            models_dir=rf"{MODELS_DIR}/models_by_epoch/{savename}",
            savename=savename,
            datasets=all_years_datasets[2022],
            tiles=tiles,
            size=image_size,
            resizing_size=resizing_size,
            n_epochs=n_epochs,
            n_bands=nbands,
            stacked_images=stacked_images,
            generate=True,
            subset="test",
        )

    # if generate_grid:
    #     print("Generando predicciones...")
    #     # Generate gridded predictions & plot examples
    #     for year in [2013, 2018, 2022]:  # all_years_datasets.keys():
    #         grid_preds = grid_predictions.generate_grid(
    #             savename,
    #             all_years_datasets,
    #             all_years_extents,
    #             image_size,
    #             resizing_size,
    #             nbands,
    #             stacked_images,
    #             year=year,
    #             generate=True,
    #         )
    #         if year==2013:
    #             grid_preds_2013 =  grid_preds
    #         grid_predictions.plot_all_examples(
    #             all_years_datasets, all_years_extents, grid_preds, grid_preds_2013, savename, year
    #         )


if __name__ == "__main__":

    variable = "avg_hh_income"

    # Selection of parameters
    params = {
        "model_name": "effnet_v2B1",
        "kind": "reg",
        "weights": None,
        "image_size": 128,
        "resizing_size": 128,
        "tiles": 1,
        "nbands": 4,
        "stacked_images": [1, 4],
        "sample_size": 1,
        "small_sample": False,
        "n_epochs": 500,
        "learning_rate": 0.0005,
        "sat_data": "aerial",
        "years": [2016, 2018, 2020, 2022, 2024], # Only the data inside WSL! all data is: [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "extra": "_Pooling",
    }

    # Run full pipeline
    run(params, train=False, compute_loss=True, generate_grid=False)
