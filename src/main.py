##############      Configuración      ##############
import os
import shutil
from xml.parsers.expat import model
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
    TRAIN_BATCH_SIZE = 8 #128
    
    # Shapes
    OUTPUT_SHAPE_IMG = (None, resizing_size, resizing_size, nbands * len(stacked_images))
    OUTPUT_SHAPE_LBL = (None,)

    # --- 1. The Python Data Loader (Slow Logic) ---
    def get_mini_batch_data(batch_indices, df_subset):
        indices = batch_indices.numpy()
        batch_imgs = []
        batch_lbls = []
        
        # Randomly pick a year for this batch to optimize Zarr access
        
        total_bands = nbands * len(stacked_images)
        target_shape = (total_bands, image_size, image_size)

        for i in indices:
            # ... (Standard extraction logic) ...
            try:
                batch_year = random.choice(available_years)
                primary_dataset = all_years_datasets[batch_year]
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
        tf.keras.layers.RandomRotation(0.25),
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
        # Validation/Test logic remains the same (Static cache)
        if not is_train:
            # ... (Keep your existing validation logic) ...
            pass 

        # --- TRAIN LOGIC: SLIDING WINDOW ---
        
        # Configuration
        ACTIVE_CYCLES = 1          # Number of files active at once (The Window Width)
        EPOCHS_TO_SURVIVE = 50000      # How long a standard file lives
        TOTAL_CACHE_SLOTS = 20     # File naming pool
        
        def generator_func(cycle_index):
            # 1. Determine Cache Lifecycle (The Stagger Logic)
            # If this is one of the very first files (0-4), we give it a shorter life
            # to ensure they don't all expire at the same time.
            # File 0 -> Lives 1 epoch  (Dies after Ep 1)
            # File 1 -> Lives 2 epochs (Dies after Ep 2)
            # ...
            # File 5+ -> Lives 5 epochs
            
            # We use int64 for comparison to match cycle_index tensor type
            idx_chk = tf.cast(cycle_index, tf.int64)
            active_chk = tf.cast(ACTIVE_CYCLES, tf.int64)
            
            repeats = tf.cond(
                idx_chk < active_chk,
                lambda: idx_chk + 1,        # Warmup phase: Staggered death
                lambda: tf.cast(EPOCHS_TO_SURVIVE, tf.int64) # Stable phase: Full life
            )

            # 2. File Naming (Cyclic slots 0-19)
            file_slot = tf.cast(cycle_index % TOTAL_CACHE_SLOTS, tf.int32)
            cache_filename = tf.strings.join([
                str(CACHE_DIR), f"/{savename}_{subset_name}_slot_", tf.strings.as_string(file_slot), ".tfcache"
            ])
            
            # 3. Clean old file
            _ = tf.py_function(clean_cache_file, [cache_filename], tf.int32)
            
            # 4. Generate Data (Slow part, runs once per file lifecycle)
            # Standardize size to avoid sync issues. 
            # CACHE_SIZE should be approx (Total_Train_Size / 5).
            all_indices_ds = tf.data.Dataset.from_tensor_slices(df_subset.index.to_numpy())
            sampled_indices_ds = all_indices_ds.shuffle(
                len(df_subset), 
                seed=tf.cast(cycle_index, tf.int64),
                reshuffle_each_iteration=False
            ).take(CACHE_SIZE) 
            
            ds = sampled_indices_ds.batch(READ_BATCH_SIZE)
            ds = ds.map(
                lambda indices: tf.py_function(
                    func=lambda x: get_mini_batch_data(x, df_subset.loc[x.numpy()]),
                    inp=[indices],
                    Tout=[tf.uint8, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            def set_shapes(imgs, lbls):
                imgs.set_shape(OUTPUT_SHAPE_IMG)
                lbls.set_shape(OUTPUT_SHAPE_LBL)
                return imgs, lbls
            
            ds = ds.map(set_shapes).unbatch()
            
            # 5. Save to Cache
            ds = ds.cache(tf.strings.as_string(cache_filename))
            
            # 6. Repeat based on the Stagger Logic
            ds = ds.repeat(repeats)
            
            return ds

        # --- THE ENGINE ---
        
        # We start 5 workers immediately. 
        # Note: The very first step of training will take time as it generates 5 files.
        master_ds = tf.data.Dataset.range(100000) \
            .interleave(
                generator_func,
                cycle_length=ACTIVE_CYCLES, # Keep 5 files open
                block_length=1,             # Take 1 batch from A, 1 from B... (Mix perfectly)
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
            )

        # 7. Standard Pipeline
        master_ds = master_ds.shuffle(1000) 
        master_ds = master_ds.batch(TRAIN_BATCH_SIZE)
        # master_ds = master_ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
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
        f"{MODELS_DIR}/models_by_epoch/{savename}/history.csv", append=True
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
    retrain: bool = False,
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

    def get_last_trained_epoch(savename):
        model_dir = MODELS_DIR / "models_by_epoch" / f"{savename}"
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            epochs = [file.split("_")[-1].replace(".keras", "") for file in files]
            epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
            
            if epochs:
                # Return the maximum epoch found
                return max(epochs)
            else:
                print("Model not found, running from beginning")
                return None
        else:
            os.makedirs(model_dir)
            print("Model not found, running from beginning")
            return None

    initial_epoch = get_last_trained_epoch(savename)
    if retrain:
        initial_epoch = None

    if initial_epoch is None:
        # constructs the model and compiles it
        model = model_function
        # optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)
        initial_epoch = 0

    else:
        print("Restoring model...")
        try:
            model_path = (
                f"{MODELS_DIR}/models_by_epoch/{savename}/{savename}_{initial_epoch}.keras"
            )
            model = keras.models.load_model(model_path)  # load the model from file
        except:
            initial_epoch -= 1
            model_path = (
                f"{MODELS_DIR}/models_by_epoch/{savename}/{savename}_{initial_epoch}.keras"
            )
            model = keras.models.load_model(model_path)  # load the model from file
        initial_epoch = initial_epoch + 1

    # The number of steps is the number of samples divided by batch size
    validation_steps = 100 # Use your TRAIN_BATCH_SIZE
    steps_per_epoch = CACHE_SIZE // 128
    model.summary()

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
    weights = params["weights"]

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
    model_name: str, kind: str, resizing_size: int, bands: int = 4, weights: str = None
):
        
    # Diccionario de modelos
    get_model_from_name = {
        "small_cnn": custom_models.small_cnn(resizing_size),  # kind=kind),
        "dinov2_model": custom_models.dinov2_model(
            resizing_size, bands=3, head="image_only", n_covariates=0, freeze_dino=False,
        ),

        "mobnet_v3_large": custom_models.mobnet_v3_large(
            resizing_size, bands=bands, kind=kind,
        ),
        "effnet_v2S": custom_models.efficientnet_v2S(
            resizing_size, bands=bands, kind=kind,
        ),
        "effnet_v2M": custom_models.efficientnet_v2M(
            resizing_size, bands=bands, kind=kind,
        ),
        "effnet_v2B1": custom_models.efficientnet_v2B1(
            resizing_size, bands=bands, kind=kind,
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

    # Load weights if provided
    if weights is not None and weights not in ["imagenet"]:
        model.load_weights(weights)
        print(f"\n--- 🚀 Successfully loaded custom weights from: {weights} ---")


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

def get_image_wrapper(idx, gdf, datasets, dataset_col, config):
    """
    Worker function that fetches one image. 
    We pass 'idx' and look up the data in the global/passed variables.
    """
    # Convert tensor to numpy int
    idx = int(idx)
    row = gdf.iloc[idx]
    
    # Default placeholder (black image)
    placeholder = np.zeros((config['resizing_size'], config['resizing_size'], config['nbands']), dtype=np.float32)

    dataset_name = row[dataset_col]
    if pd.isna(dataset_name) or dataset_name not in datasets:
        return placeholder

    ds = datasets[dataset_name]
    geom = row.geometry

    # Handle Point vs Polygon
    if geom.geom_type != 'Point':
        point = (geom.centroid.x, geom.centroid.y)
        polygon = geom
    else:
        point = (geom.x, geom.y)
        polygon = geom.buffer(10)

    try:
        # Heavy IO operation
        image, _ = geo_utils.stacked_image_from_census_tract(
            dataset=ds,
            polygon=polygon,
            point=point,
            img_size=config['image_size'],
            n_bands=config['nbands'],
            stacked_images=config['stacked_images'],
            bounds=True
        )
        
        # Preprocessing
        processed_img = geo_utils.process_image(
            image, 
            resizing_size=config['resizing_size'], 
            moveaxis=True 
        )
        return processed_img.astype(np.float32)
        
    except Exception:
        return placeholder


def predict_buildings_income(
    model, 
    year=2022, 
    batch_size=128,  # Increased batch size for GPU efficiency
    image_size=128, 
    resizing_size=128, 
    nbands=4, 
    stacked_images=[1],
    output_name="predictions"
):
    from tqdm import tqdm 

    # --- 2. Setup Data (Main Thread) ---
    # 1. Load the Building Data
    gdf = build_dataset.load_income_dataset()

    print(f"Loading Satellite Imagery for {year}...")
    datasets, extents = build_dataset.load_satellite_datasets(year=year)
    
    print("Mapping buildings to tiles...")
    gdf = build_dataset.assign_datasets_to_gdf(
        gdf, extents, year=year, centroid=True, buffer=False
    )

    dataset_col = f"dataset_{year}"
    
    # Configuration dict to pass to workers
    config = {
        'image_size': image_size,
        'resizing_size': resizing_size,
        'nbands': nbands,
        'stacked_images': stacked_images
    }

    # --- 3. Build the TF.Data Pipeline ---
    print("Building Parallel Pipeline...")
    
    # Create a dataset of INDICES (0, 1, 2, ... N)
    indices_ds = tf.data.Dataset.range(len(gdf))

    def map_func(idx):
        img = tf.py_function(
            func=lambda i: get_image_wrapper(i, gdf, datasets, dataset_col, config),
            inp=[idx],
            Tout=tf.float32
        )
        # Explicitly set shape so TensorFlow knows the dimensions
        img.set_shape([resizing_size, resizing_size, nbands * len(stacked_images)])
        return idx, img

    # Apply map and batching
    dataset = indices_ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Define augmentation layer
    aug_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomContrast(0.2), # <--- This is likely the key factor
    ])


    # --- Setup Output File ---
    output_file = RESULTS_DIR / f"{year}_{output_name}.csv"
    # Create file with headers (overwrites if exists)
    pd.DataFrame(columns=['index', 'predicted_value']).to_csv(output_file, index=False)
    print(f"Saving incremental progress to: {output_file}")

    # --- Start Iteration ---
    print(f"Starting prediction on {len(gdf)} buildings...")
    total_batches = int(np.ceil(len(gdf) / batch_size))

    # Manually iterate over the dataset
    for batch_indices, batch_images in tqdm(dataset, total=total_batches, desc="Predicting"):
        
        # 1. Predict on the current batch (runs on GPU)
        # batch_images = aug_layer(batch_images, training=True) # Force the transformation
        batch_preds = model.predict_on_batch(batch_images)
        
        # 2. Convert to numpy and flatten
        ids = batch_indices.numpy()
        vals = batch_preds.flatten()
        
        # 3. Create a temporary DataFrame
        batch_df = pd.DataFrame({
            'index': ids, 
            'predicted_value': vals
        })
        
        # 4. Append to CSV immediately
        # mode='a' appends, header=False avoids repeating headers
        batch_df.to_csv(output_file, mode='a', header=False, index=False)

    print("Prediction complete!")    

    return gdf

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
    generate_predictions=False,
    retrain=False,
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

    savename = generate_savename(
        model_name, image_size, learning_rate, stacked_images, years, extra
    )
    log_dir = f"{LOGS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    generate_parameters_log(params, savename)
    
    # print("**"*10)
    # print("CORRIENDO SOLO CON 2022 PARA TESTEAR. SI TE APARECE ESTO, REVISAR!!!")
    # print("**"*10)
    years = [2022]
    all_years_datasets, all_years_extents, df = open_datasets(
        sat_data=sat_data, years=years
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
            lr=0.00005,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss=loss,
            metrics=metrics,
            callbacks=callbacks,
            epochs=n_epochs,
            savename=savename,
            logdir=log_dir,
            retrain=retrain,
        )
        print("Fin del entrenamiento")

    ## Compute metrics
    # Genero la test_loss por RC
    if compute_loss:
        true_metrics.compute_loss(  # No entra el test_dataset acá pero despues usa el df_test guardado en memoria
            models_dir=rf"{MODELS_DIR}/models_by_epoch/{savename}",
            savename=savename,
            datasets=all_years_datasets,
            tiles=tiles,
            size=image_size,
            resizing_size=resizing_size,
            n_epochs=n_epochs,
            n_bands=nbands,
            stacked_images=stacked_images,
            generate=True,
            subset="test",
        )

    if generate_predictions:
        print("Generando predicciones...")
        model = tf.keras.models.load_model(MODELS_DIR / f"{savename}.keras", compile=False)  # Load the best model saved during training
        df_result = predict_buildings_income(
            model=model,
            image_size=image_size, 
            resizing_size=resizing_size, 
            nbands=nbands, 
            stacked_images=stacked_images,
            year=2022, 
            output_name=f"testing_predictions_{savename}"
        )
        df_result.to_parquet( RESULTS_DIR / "nyc_buildings_with_predictions.parquet")

        df = pd.read_csv(RESULTS_DIR / f"2022_predictions_{savename}.csv")
        gdf = build_dataset.load_income_dataset()
        gdf = gdf.join(df, how="inner")
        gdf.to_parquet(RESULTS_DIR / f"2022_predictions_{savename}.parquet")

        # Fix broken geometries and dissolve by census tract to get tract-level predictions
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf_by_census_tract = gdf.dissolve(
            "GEOID", 
            aggfunc={'var': 'mean', 'predicted_value': 'mean'}
        )
        gdf_by_census_tract.to_parquet(RESULTS_DIR / f"2022_predictions_by_tract_{savename}.parquet")

        # # Generate gridded predictions & plot examples
        # for year in [2022]:  # all_years_datasets.keys():
        #     grid_preds = grid_predictions.generate_grid(
        #         savename,
        #         all_years_datasets,
        #         all_years_extents,
        #         image_size,
        #         resizing_size,
        #         nbands,
        #         stacked_images,
        #         year=year,
        #         generate=True,
        #     )
        #     if year==2013:
        #         grid_preds_2013 =  grid_preds
        #     grid_predictions.plot_all_examples(
        #         all_years_datasets, all_years_extents, grid_preds, grid_preds_2013, savename, year
        #     )


if __name__ == "__main__":

    variable = "avg_hh_income"

    # Selection of parameters
    params = {
        "model_name": "dinov2_model",
        "kind": "reg",
        "weights": None,
        "image_size": 224,
        "resizing_size": 224,
        "tiles": 1,
        "nbands": 3,
        "stacked_images": [1],
        "sample_size": 1,
        "small_sample": False,
        "n_epochs": 100,
        "learning_rate": 0.001,
        "sat_data": "aerial",
        "years": [2022], # Only the data inside WSL! all data is: [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "extra": "",  # Extra info to add to the savename (e.g. for ablation studies)
    }

    # Run full pipeline
    run(params, train=True, retrain=True, compute_loss=False, generate_predictions=False)
