##############      Configuración      ##############
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src import custom_models_tf
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR, IMAGERY_ROOT, MODELS_DIR
pd.set_option("display.max_columns", None)

# path_programas  = globales[7]
###############################################

import src.main as main
import src.true_metrics as true_metrics

import os
import sys
import scipy
import random
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


def get_true_val_loss(params):

    params = main.fill_params_defaults(params)

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

    savename = main.generate_savename(
        model_name, image_size, learning_rate, stacked_images, years, extra
    )
    metrics_path = rf"{MODELS_DIR}/models_by_epoch/{savename}/test_metrics_over_epochs.csv"
    if not os.path.exists(metrics_path):

        all_years_datasets, all_years_extents, df = main.open_datasets(
            sat_data=sat_data, years=[2013]
        )

        metrics_epochs = true_metrics.compute_custom_loss_all_epochs(
            rf"{MODELS_DIR}/models_by_epoch/{savename}",
            savename,
            all_years_datasets[
                2013
            ],  # Only 2013 because I want to test with the ground truth images...
            tiles,
            image_size,
            resizing_size,
            "test",
            n_epochs,
            nbands,
            stacked_images,
            generate=False,
            verbose=True,
        )

    df = pd.read_csv(
        metrics_path,
        index_col="epoch",
        usecols=["epoch", "mse_train", "mse_test_rc"],
        nrows=n_epochs,
    )

    return df


def compute_experiment_results(options, experiment_name):
    """Compute true loss if needed and plot comparison between the different options"""
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
    data_for_plot = {}

    for name, params in options.items():

        df = get_true_val_loss(params)

        sns.lineplot(
            df["mse_train"].ewm(span=5, adjust=False).mean(),
            ax=ax[0],
            label=f"{name}",
            legend=True,
        )
        sns.lineplot(
            df["mse_test_rc"].ewm(span=5, adjust=False).mean(),
            ax=ax[1],
            label=f"{name}",
            legend=False,
        )
        data_for_plot[name] = df

    ax[0].set_ylim(0, 0.3)
    ax[0].set_ylabel("")
    ax[0].set_title("ECM Entrenamiento")
    ax[0].set_xlabel("Época")
    sns.despine(ax=ax[0])

    ax[1].set_ylim(0, 0.3)
    ax[1].set_ylabel("")
    ax[1].set_title("ECM Conjunto de Prueba")
    ax[1].set_xlabel("Época")
    sns.despine(ax=ax[1])
    sns.move_legend(ax[0], "lower left")

    plt.savefig(rf"{RESULTS_DIR}/{experiment_name}.png", dpi=300, bbox_inches="tight")
    print("Se creó la imagen " + rf"{RESULTS_DIR}/{experiment_name}.png")


if __name__ == "__main__":
    import warnings

    experiment_name = "learning_rate"
    options = {
        "lr=0.001": {"learning_rate": 0.001, "n_epochs": 100},
        "lr=0.0001": {"learning_rate": 0.0001, "n_epochs": 100},
        "lr=0.00001": {"learning_rate": 0.00001, "n_epochs": 100},
    }
    compute_experiment_results(options, experiment_name)

    experiment_name = "models"
    options = {
        "EfficientNetV2 S": {"model_name": "effnet_v2S", "n_epochs": 100},
        "EfficientNetV2 M": {"model_name": "effnet_v2M", "n_epochs": 100},
        "EfficientNetV2 L": {"model_name": "effnet_v2L", "n_epochs": 100},
    }
    compute_experiment_results(options, experiment_name)

    experiment_name = "años_utilizados"
    options = {
        "2013": {"years": [2013], "n_epochs": 100},
        "2013 y 2018": {"years": [2013, 2018], "n_epochs": 100},
        "2013, 2018 y 2022": {
            "years": [2013, 2018, 2022],
            "n_epochs": 100,
        },
    }

    compute_experiment_results(options, experiment_name)

    experiment_name = "nbands"
    options = {
        "RGB": {
            "nbands": 3,
            "extra": "_RGBonly",
            "years": [2013, 2018, 2022],
            "n_epochs": 100,
        },
        "RGB+NIR": {"nbands": 4, "years": [2013, 2018, 2022], "n_epochs": 100},
    }
    compute_experiment_results(options, experiment_name)

    # experiment_name = "img_size"
    # options = {
    #     "50x50mts": {
    #         "image_size": 128,
    #         "years": [2013, 2018, 2022],
    #         "stacked_images": [1],
    #         "n_epochs": 150,
    #     },
    #     "100x100mts": {
    #         "image_size": 256,
    #         "years": [2013, 2018, 2022],
    #         "stacked_images": [1],
    #         "n_epochs": 150,
    #     },
    #     "200x200mts": {
    #         "image_size": 512,
    #         "years": [2013, 2018, 2022],
    #         "stacked_images": [1],
    #         "n_epochs": 150,
    #     },
    #     "50x50mts + 100x100mts": {
    #         "image_size": 128,
    #         "years": [2013, 2018, 2022],
    #         "stacked_images": [1, 2],
    #         "n_epochs": 150,
    #     },
    #     "50x50mts + 200x200mts": {
    #         "image_size": 128,
    #         "years": [2013, 2018, 2022],
    #         "stacked_images": [1, 4],
    #         "n_epochs": 150,
    #     },
    # }
    # compute_experiment_results(options, experiment_name)
