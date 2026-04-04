##############      Configuración      ##############
from ast import Return
import os
import pickle
from tokenize import String
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR, IMAGERY_ROOT
import warnings

pd.set_option("display.max_columns", None)

# path_programas  = globales[7]

import geopandas as gpd
import xarray as xr
import shapely.geometry as sg
import pandas as pd
import src.geo_utils as geo_utils


def load_satellite_datasets(year=2014, stretch=False, engine="zarr"):
    """Load satellite datasets and get their extents"""

    if engine=="zarr":
        file = f"nyc_{year}.zarr"
        dataset_path = IMAGERY_ROOT
        files = [file]

    elif engine=="tif":
        dataset_path = IMAGERY_ROOT / year
        files = os.listdir(dataset_path)
        files = [f for f in files if f.endswith(".tif")]
        assert all([os.path.isfile(dataset_path / f) for f in files])

    else:
        raise ValueError(f"Unknown engine: {engine}. Valid engines are: zarr, tif")

    print(f"Loading {len(files)} files from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise ValueError(f"Year {year} images not found: {dataset_path} does exist! Check they are stored in WSL!")
    datasets = {
        f: (filter_black_pixels(xr.open_dataset(dataset_path / f, engine=engine,  mask_and_scale=False)))
        for f in files
    }

    if stretch:
        datasets = {name: stretch_dataset(ds) for name, ds in datasets.items()}

    extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}

    return datasets, extents


# def load_landsat_datasets(stretch=False):
#     """Load satellite datasets and get their extents"""

#     files = os.listdir(rf"{path_landsat}")
#     assert os.path.isdir(rf"{path_landsat}")
#     files = [f for f in files if f.endswith(".tif")]
#     assert all([os.path.isfile(rf"{path_landsat}/{f}") for f in files])

#     datasets = {
#         f.replace(".tif", ""): (
#             normalize_landsat(xr.open_dataset(rf"{path_landsat}/{f}"))
#         )
#         for f in files
#     }
#     if stretch:
#         datasets = {name: stretch_dataset(ds) for name, ds in datasets.items()}

#     extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}

#     return datasets, extents


# def load_nightlight_datasets(stretch=False):
#     """Load satellite datasets and get their extents"""

#     files = os.listdir(rf"{path_nocturnas}")
#     assert os.path.isdir(rf"{path_nocturnas}")
#     files = [f for f in files if f.endswith(".tif")]
#     assert all([os.path.isfile(rf"{path_nocturnas}/{f}") for f in files])

#     datasets = {
#         f.replace(".tif", ""): (xr.open_dataset(rf"{path_nocturnas}/{f}"))
#         for f in files
#     }
#     if stretch:
#         datasets = {name: stretch_dataset(ds) for name, ds in datasets.items()}

#     extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}

#     return datasets, extents


def load_income_dataset(variable="avg_hh_inc", trim=False, log=True):
    """
    Load and normalize income dataset. Returns split temporal and spatial data.
    
    Temporal data (Long Format):
    - GEOID, var (normalized income), type (train/val/test - added later)
    
    Spatial data (Cross-Sectional):
    - GEOID, geometry
    
    This split allows efficient caching: geometries stay on disk until needed,
    temporal data is lightweight and filters quickly.
    
    Returns:
    --------
    df_temporal : pd.DataFrame with columns [GEOID, var]
    gdf_spatial : gpd.GeoDataFrame with columns [GEOID, geometry]
    """

    # Open SAE dataset
    print("Loading income dataset...")
    gdf = gpd.read_parquet(PROCESSED_DATA_DIR / "nyc_buildings_with_sae.parquet")
    
    # id_building is the original unique index of the buildings GeoDataFrame
    gdf["id_building"] = gdf.index
    gdf["id_building_year"] = gdf["id_building"].astype(str) + "_" + gdf["year"].astype(str)

    # FIXME: Probably it's a good idea to reproject all the zarr so I can scale this to anywhere...
    gdf = gdf.to_crs("EPSG:6539")
    gdf = gdf[gdf[variable]>0]  # Remove ct with no income data 

    # FIXME: I forgot to add Bronx for some reason... Should add it later
    gdf = gdf[gdf[variable]>0]

    if trim:
        # Should not be needed with these models
        gdf = gdf[gdf["Area"] <= 200000]  # Remove ct that are too big
    gdf = gdf.reset_index(drop=True)

    # Normalize ELL estimation:
    # FIXME: I should do this normalization based on census tract estimates, not buildings...
    if log:
        gdf[variable + "_log"] = gdf[variable].apply(lambda x: np.log(x))
        variable = variable + "_log"

    var_mean = gdf[variable].mean()
    var_std = gdf[variable].std()
    gdf["var"] = (gdf[variable] - var_mean) / var_std

    date = pd.Timestamp.now().strftime("%Y%m%d")
    data_dict = {"mean": var_mean, "std": var_std}
    pd.DataFrame().from_dict(data_dict, orient="index", columns=[variable]).to_csv(
        PROCESSED_DATA_DIR / f"scalars_{variable}_trim{trim}_{date}.csv"
    )

    # ======== SPLIT INTO TEMPORAL AND SPATIAL DATA ========
    
    # 1. TEMPORAL DATA (Long Format): [GEOID, var] - lightweight, fast to filter
    df_temporal = gdf[["id_building_year", "id_building", "GEOID", "year", "var"]].copy()
    
    # 2. SPATIAL DATA (Cross-Sectional): [GEOID, geometry] - indexed by GEOID for fast joins
    gdf_spatial = gdf[["id_building", "geometry"]].drop_duplicates("id_building").copy()
    gdf_spatial = gdf_spatial.set_index("id_building")
    
    # Save both to Parquet
    temporal_path = PROCESSED_DATA_DIR / f"temporal_data_{variable}_trim{trim}_{date}.parquet"
    spatial_path = PROCESSED_DATA_DIR / f"geometries_{variable}_trim{trim}_{date}.parquet"
    
    df_temporal.to_parquet(temporal_path)
    gdf_spatial.to_parquet(spatial_path)
    
    print(f"Saved temporal data: {temporal_path}")
    print(f"Saved spatial data: {spatial_path}")
    
    # Also keep the old combined format for backward compatibility
    gdf.to_parquet(PROCESSED_DATA_DIR / f"gdf_{variable}_trim{trim}_{date}.parquet")
    
    return df_temporal, gdf_spatial


def load_temporal_data(variable="avg_hh_inc", trim=False, log=True):
    """
    Load only the temporal (non-spatial) data.
    Returns: pd.DataFrame with [GEOID, var]
    """
    date_pattern = "*"  # Find the most recent file
    files = list(PROCESSED_DATA_DIR.glob(f"temporal_data_{variable}_trim{trim}_{date_pattern}.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No temporal data files found for variable={variable}, trim={trim}")
    
    # Load the most recent file
    latest_file = sorted(files)[-1]
    print(f"Loading temporal data from: {latest_file}")
    return pd.read_parquet(latest_file)


def load_geometries(variable="avg_hh_inc", trim=False, log=True):
    """
    Load only the spatial (geometry) data.
    Returns: gpd.GeoDataFrame with [GEOID, geometry] (indexed by GEOID)
    """
    date_pattern = "*"  # Find the most recent file
    files = list(PROCESSED_DATA_DIR.glob(f"geometries_{variable}_trim{trim}_{date_pattern}.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No geometry files found for variable={variable}, trim={trim}")
    
    # Load the most recent file
    latest_file = sorted(files)[-1]
    print(f"Loading geometries from: {latest_file}")
    return gpd.read_parquet(latest_file)


def lazy_join_geometries(df_temporal, gdf_spatial, id_column="id_building"):
    """
    Perform a lazy join of temporal data with spatial geometries.
    Only called when geometries are actually needed (e.g., during validation or final output).
    
    Parameters:
    -----------
    df_temporal : pd.DataFrame with GEOID column
    gdf_spatial : gpd.GeoDataFrame indexed by GEOID
    id_column : str, name of the GEOID column in df_temporal
    
    Returns:
    --------
    gdf : gpd.GeoDataFrame with both temporal and spatial data
    """
    gdf = gpd.GeoDataFrame(
        df_temporal.set_index(id_column),
        geometry=gdf_spatial.loc[df_temporal[id_column], "geometry"].values,
        crs=gdf_spatial.crs
    )
    return gdf


def get_spatial_image_mapping(
    gdf_shapes,
    extents,
    year,
    centroid=False,
    buffer=True,
    select="first_match",
    verbose=True,
):
    """
    Spatially intersects geometries with image extents for a specific year.
    Returns a mapping dictionary: GEOID -> dataset_name (for a single year).
    
    This is designed to be called once per year, then the results are merged into
    the temporal DataFrame as a new column f"dataset_{year}".
    
    Parameters:
    -----------
    gdf_shapes : gpd.GeoDataFrame with [GEOID, geometry]
    extents : dict mapping dataset_name -> bounding box polygon
    year : int, the year being processed
    centroid : bool, use centroids instead of full geometry
    buffer : bool, buffer geometries before intersection
    select : str, "first_match" or "all_matches"
    verbose : bool, print statistics
    
    Returns:
    --------
    pd.Series indexed by GEOID with dataset_name values (or NaN if no match)
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Work on a lightweight copy so we don't pollute the master geometries
    temp_gdf = gdf_shapes.reset_index()[["id_building", "geometry"]].copy()
    temp_gdf["dataset"] = np.nan

    if centroid:
        temp_gdf["geometry"] = temp_gdf.centroid

    temp_gdf["dataset"] = np.nan

    if select == "first_match":
        for name, bbox in extents.items():
            if buffer:
                temp_gdf.loc[temp_gdf.buffer(0.004).within(bbox), "dataset"] = name
            else:
                temp_gdf.loc[temp_gdf.within(bbox), "dataset"] = name

    elif select == "all_matches":
        def get_matching_names(row):
            return [name for name in extents.keys() if row[name] is True]

        for name, bbox in extents.items():
            if buffer:
                temp_gdf[name] = temp_gdf.buffer(0.004).intersects(bbox)
            else:
                temp_gdf[name] = temp_gdf.intersects(bbox)

        temp_gdf["dataset"] = temp_gdf.apply(get_matching_names, axis=1)
        temp_gdf["dataset"] = temp_gdf["dataset"].apply(lambda x: x if len(x) > 0 else np.nan)
        temp_gdf = temp_gdf.drop(columns=list(extents.keys()))

    if verbose:
        nan_links = temp_gdf["dataset"].isna().sum()
        total_links = len(temp_gdf)
        print(f"[{year}] Geometries without images: {nan_links} out of {total_links}")

    warnings.filterwarnings("default")

    # Return a Series indexed by GEOID (for easy merging into temporal data)
    return temp_gdf.set_index("id_building")["dataset"]


def get_test_area_from_file(filename = "Test_NYC_Area.parquet"):
    test = gpd.read_parquet(RAW_DATA_DIR / filename)
    test_polygon = test.dissolve().geometry.iloc[0]
    return test_polygon

def split_train_test(df_temporal, train_years, test_years, gdf_spatial=None, buffer=500):
    """
    Splits the temporal DataFrame into 'train' and 'test' based on spatial and temporal criteria.
    
    The temporal data must have been previously merged with spatial data and a 'year' column
    added via get_spatial_image_mapping().
    
    Logic:
    - TEST:  year is in test_years OR geometry is strictly INSIDE the test_polygon.
    - TRAIN: year is in train_years AND geometry is strictly OUTSIDE the (test_polygon + buffer).
    - DROP:  Geometry overlaps the border or falls within the buffer zone.
    
    Parameters:
    -----------
    df_temporal : pd.DataFrame with columns [GEOID, var, year, dataset_year]
    train_years : list of years for training
    test_years : list of years for testing
    gdf_spatial : gpd.GeoDataFrame (optional), indexed by GEOID. If provided, used for spatial filtering.
    buffer : int, buffer distance in meters for the exclusion zone
    
    Returns:
    --------
    df_temporal : pd.DataFrame with added 'type' column [train/test/None]
    """
    
    test_polygon = get_test_area_from_file()
    
    # Initialize column with NaNs
    df_temporal["type"] = np.nan
    
    # ====== TEMPORAL FILTERING ======
    # 3- Include all rows from test_years in the test set 
    test_year_mask = df_temporal["year"].isin(test_years)
    df_temporal.loc[test_year_mask, "type"] = "test"
    
    train_year_mask = df_temporal["year"].isin(train_years)
    
    # ====== SPATIAL FILTERING (only if gdf_spatial provided) ======
    if gdf_spatial is not None:
        # For each GEOID, check if it falls in test area or buffer zone
        # We can group by GEOID since spatial properties are constant per building
        
        geoid_spatial_type = {}
        for geoid in df_temporal["id_building"].unique():
            if geoid not in gdf_spatial.index:
                geoid_spatial_type[geoid] = None  # Drop if no geometry
                continue
            
            geometry = gdf_spatial.loc[geoid, "geometry"]
            
            # CHECK TEST: Must be strictly inside the test polygon
            if geometry.within(test_polygon):
                geoid_spatial_type[geoid] = "test_spatial"
            
            # CHECK TRAIN: Must be strictly outside the (test polygon + buffer)
            elif geometry.disjoint(test_polygon.buffer(buffer)):
                geoid_spatial_type[geoid] = "train_spatial"
            
            else:
                # Falls in buffer zone - will be dropped
                geoid_spatial_type[geoid] = None
        
        # Apply spatial type to all rows with that GEOID
        for geoid, spatial_type in geoid_spatial_type.items():
            geoid_mask = df_temporal["id_building"] == geoid
            
            if spatial_type == "test_spatial":
                df_temporal.loc[geoid_mask, "type"] = "test"
            elif spatial_type == "train_spatial" and train_year_mask[geoid_mask].any():
                # Only mark as train if it also has a training year
                df_temporal.loc[geoid_mask & train_year_mask, "type"] = "train"
            else:
                # Mark for deletion (buffer zone)
                df_temporal.loc[geoid_mask, "type"] = np.nan
    else:
        # No spatial filtering - just use temporal split
        df_temporal.loc[train_year_mask, "type"] = "train"
    
    # ====== CALCULATE STATS ======
    test_size = df_temporal[df_temporal["type"] == "test"].shape[0]
    train_size = df_temporal[df_temporal["type"] == "train"].shape[0]
    invalid_size = df_temporal[df_temporal["type"].isna()].shape[0]
    total_size = df_temporal.shape[0]

    print(
        "\n",
        f"Size of test dataset: {test_size/total_size*100:.2f}% ({test_size} rows)",
        f"Size of train dataset: {train_size/total_size*100:.2f}% ({train_size} rows)",
        f"Deleted rows due to train/test overlapping: {invalid_size/total_size*100:.2f}% ({invalid_size} rows)",
        sep="\n",
    )

    return df_temporal

from shapely.geometry import box

def assert_train_test_datapoint(bounds, test_polygon, wanted_type="train", buffer=500):
    """
    Returns True if the datapoint (defined by bounds) matches the wanted_type 
    relative to the test_polygon.
    
    Parameters:
    - bounds: A tuple/list of (min_x, min_y, max_x, max_y)
    - test_polygon: The Shapely polygon defining the test area.
    - wanted_type: "train" or "test"
    - buffer: The safety buffer distance (must match what you used in splitting).
    """
    
    # 1. Convert the bounds tuple into a Shapely geometry object
    #    The * unpacks the tuple (minx, miny, maxx, maxy)
    datapoint_geom = box(*bounds)
    
    # 2. Determine the actual type of this datapoint
    actual_type = None

    # CHECK TEST: Must be strictly inside the test polygon
    if datapoint_geom.within(test_polygon):
        actual_type = "test"
        
    # CHECK TRAIN: Must be strictly outside the (test polygon + buffer)
    # We apply the buffer here to ensure we respect the exclusion zone
    elif datapoint_geom.disjoint(test_polygon.buffer(buffer)):
        actual_type = "train"
        
    # (If it's neither, actual_type remains None, representing the buffer zone)

    # 3. Assert
    return actual_type == wanted_type


def get_dataset_for_gdf(gdf, datasets, link, year=2013, id_var="GEOID"):
    """Get dataset where the census tract is located."""
    
    # 1. Get all matches as a Series (do not squeeze)
    matches = gdf.loc[gdf[id_var] == link, f"dataset_{year}"]

    # 2. Check if we found anything
    if matches.empty:
        return None

    # 3. Take the first match. 
    # Whether there is 1 row or 100 duplicates, this safely gets the first string.
    current_ds_name = matches.iloc[0]

    # 4. Handle NaNs (if the cell was empty)
    if pd.isna(current_ds_name):
        return None

    # 5. Return the dataset
    # using .get() is safer than brackets [], but brackets are fine if you trust your data keys
    return datasets.get(current_ds_name)

def add_buffer(bounds, buffer):
    """Add buffer to bounds.

    Parameters:
    -----------
    bounds: tuple, (minx, miny, maxx, maxy)
    buffer: int, buffer to add to bounds

    Returns:
    --------
    bounds: dict, {'minx': minx-buffer, 'miny': miny-buffer, 'maxx': maxx+buffer, 'maxy': maxy+buffer}
    """
    minx, miny, maxx, maxy = bounds
    return {
        "minx": minx - buffer,
        "miny": miny - buffer,
        "maxx": maxx + buffer,
        "maxy": maxy + buffer,
    }

def crop_dataset_to_link(ds, gdf, link):
    # obtengo el poligono correspondiente al link
    gdf_sub = gdf.loc[gdf["GEOID"] == link].copy() 
    if gdf_sub.empty:
        return None

    # Try to repair invalid geometries (common fix: buffer(0) or shapely.make_valid)
    try:

        # use unary_union (avoids groupby/dissolve topology issues)
        multipolygon = gdf_sub.union_all()

        if multipolygon is None or multipolygon.is_empty:
            return None

        if not multipolygon.is_valid:
            multipolygon = multipolygon.buffer(0)

    except Exception as e:
        # Log and skip problematic geometry (caller handles None)
        print(f"Warning: invalid geometry for link {link}: {e}")
        return None

    # Get bounds of the shapefile's polygon
    bbox_img = add_buffer(multipolygon.bounds, 1000)

    # Filter dataset based on the bounds of the shapefile's polygon
    image_ds = ds.sel(
        x=slice(float(bbox_img["minx"]), float(bbox_img["maxx"])),
        y=slice(float(bbox_img["maxy"]), float(bbox_img["miny"])),
    )
    return image_ds


def get_prediction_images_for_link(
    ds,
    gdf,
    link,
    tiles,
    size,
    resizing_size,
    sample,
    n_bands=4,
    stacked_images=[1],
):
    """
    Itera sobre el bounding box del poligono del radio censal, tomando imagenes de tamño sizexsize
    Si dicha imagen se encuentra dentro del polinogo, se genera el composite con dicha imagen mas otras tiles**2 -1 imagenes
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    gdf: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """

    images = []
    points = []
    bounds = []
    total_bands = len(stacked_images) * n_bands

    link_dataset = crop_dataset_to_link(ds, gdf, link)
    link_geometries = gdf.loc[gdf["GEOID"] == link, "geometry"].values
    for building_geometry in link_geometries:

        image_point = building_geometry.centroid
        point = image_point.coords[0]
        image, bound = geo_utils.stacked_image_from_census_tract(
            dataset=link_dataset,
            polygon=building_geometry,
            point=point,
            img_size=size,
            n_bands=n_bands,
            stacked_images=stacked_images,
        )

        if image.shape == (total_bands, size, size):
            # TODO: add a check to see if the image is contained in test bounds
            image = geo_utils.process_image(image, resizing_size)

            images += [image]
            bounds += [bound]

        else:
            print("Image failed")

    return images, points, bounds


def get_gridded_images_for_dataset(
    model, ds, gdf, tiles, size, resizing_size, bias, sample, to8bit
):
    """
    Itera sobre el bounding box de un dataset (raster de imagenes), tomando imagenes de tamño sizexsize
    Asigna el valor "real" del radio censal al que pertenece el centroide de la imagen.
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen,
    un array con los valores "reales" de los radios censales y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    gdf: geopandas.GeoDataFrame, shapefile con los radios censales
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """
    import main
    from shapely.geometry import Polygon

    # FIXME: algunos radios censales no se generan bien. Ejemplo: 065150101. ¿Que pasa ahi?
    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_predictions = np.empty((0))
    batch_real_values = np.empty((0))
    batch_bounds = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))
    all_bounds = np.empty((0))

    tile_size = size // tiles
    tiles_generated = 0

    # Iterate over the center points of each image:
    # - Start point is the center of the image (tile_size / 2, start_index)
    # - End point is the maximum possible center point (link_dataset.y.size)
    # - Step is the size of each image (tile_size)

    # FIXME: para mejorar la eficiencia, convendría hacer un dissolve de gdf y verificar que
    # image_point este en ese polygono y no en todo el df
    start_index = int(tile_size / 2)
    for idy in range(start_index, ds.y.size, tile_size):
        # Iterate over columns
        for idx in range(start_index, ds.x.size, tile_size):
            # Get the center point of the image
            image_point = (float(ds.x[idx]), float(ds.y[idy]))
            point_geom = sg.Point(image_point)

            # Get data for selected point
            radio_censal = gdf.loc[gdf.contains(point_geom)]
            if radio_censal.empty:
                # El radio censal no existe, es el medio del mar...
                continue

            real_value = radio_censal["var"].values[0]
            link_name = radio_censal["GEOID"].values[0]

            # Check if the centroid of the image is within the original polygon:
            #   - if it is, then generate the n images

            image, point, bound, tbound = geo_utils.random_image_from_census_tract(
                ds,
                gdf,
                link_name,
                start_point=image_point,
                tiles=tiles,
                size=size,
                bias=bias,
                to8bit=to8bit,
            )

            if image is not None:
                image = geo_utils.process_image(image, resizing_size)
                geom_bound = Polygon(
                    bound[0]
                )  # Create polygon of the shape of the image

                batch_images = np.concatenate([batch_images, np.array([image])], axis=0)
                batch_link_names = np.concatenate(
                    [batch_link_names, np.array([link_name])], axis=0
                )
                batch_real_values = np.concatenate(
                    [batch_real_values, np.array([real_value])], axis=0
                )
                batch_bounds = np.concatenate(
                    [batch_bounds, np.array([geom_bound])], axis=0
                )

                # predict with the model over the batch
                if batch_images.shape[0] == 128:
                    # predictions
                    batch_predictions = main.get_batch_predictions(
                        model, batch_images
                    )

                    # Store data
                    all_predictions = np.concatenate(
                        [all_predictions, batch_predictions], axis=0
                    )
                    all_link_names = np.concatenate(
                        [all_link_names, batch_link_names], axis=0
                    )
                    all_real_values = np.concatenate(
                        [all_real_values, batch_real_values], axis=0
                    )
                    all_bounds = np.concatenate([all_bounds, batch_bounds], axis=0)

                    # Restore batches to empty
                    batch_images = np.empty((0, resizing_size, resizing_size, 4))
                    batch_predictions = np.empty((0))
                    batch_link_names = np.empty((0))
                    batch_predictions = np.empty((0))
                    batch_real_values = np.empty((0))
                    batch_bounds = np.empty((0))

    # Creo dataframe para exportar:
    d = {
        "GEOID": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }

    df_preds = gpd.GeoDataFrame(d, geometry=all_bounds, crs="epsg:6539")

    return df_preds


def stretch_dataset(ds, pixel_depth=32_767):
    """Stretch band data from satellite images."""
    minimum = ds.band.quantile(0.01).values
    maximum = ds.band.quantile(0.99).values
    ds = (ds - minimum) / (maximum - minimum) * pixel_depth
    ds = ds.where(ds.band > 0, 0)
    ds = ds.where(ds.band < pixel_depth, pixel_depth)
    return ds


def normalize_landsat(ds):
    band = ds.band.to_numpy()
    for band in range(band.shape[0]):
        this_band = band[band]

        vmin = np.percentile(this_band, q=2)
        vmax = np.percentile(this_band, q=98)

        # High values
        mask = this_band > vmax
        this_band[mask] = vmax

        # low values
        mask = this_band < vmin
        this_band[mask] = vmin

        # Normalize
        this_band = (this_band - vmin) / (vmax - vmin)

        band[band] = this_band * 255

    return ds


def remove_overlapping_pixels(main, to_crop):

    main_extent = geo_utils.get_dataset_extent(main)
    will_be_cropeed_extent = geo_utils.get_dataset_extent(to_crop)
    cropped_extent = will_be_cropeed_extent.difference(main_extent)

    # Crop dataset
    min_lon, min_lat, max_lon, max_lat = cropped_extent.bounds
    cropped = to_crop.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))

    return cropped


def pickle_xr_dataset(ds, filename):
    import pickle

    pkl = pickle.dumps(ds, protocol=-1)
    with open(filename, "wb") as f:
        f.write(pkl)

    print("Pickled data saved to:", filename)
    return


def add_datasets_combinations(datasets):
    from shapely.geometry import box

    extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}
    combinations = {}
    to_remove = []

    for ds_name, ds in datasets.items():
        # Construyo lista de datasets que intersectan con ds_name
        capture_ds_name = ds_name.split("_")[1]
        ds_extent = extents[ds_name]
        buffered_extent = ds_extent.buffer(0.005).envelope
        xmin, ymin, xmax, ymax = buffered_extent.bounds

        intersecting = []
        for name, ds_extent in extents.items():
            capture_name = name.split("_")[1]
            if (
                ds_extent.intersects(buffered_extent)
                & (name != ds_name)
                & (capture_ds_name == capture_name)
            ):
                intersecting += [name]

        # Recorto datasets de intersection (buffer de 1080px):
        cropped_datasets = {}
        for intersection in intersecting:
            intersecting_ds = datasets[intersection]
            cropped_datasets[intersection] = intersecting_ds.sel(
                x=slice(xmin, xmax), y=slice(ymax, ymin)
            )

        # Armo xarray con la intersección de a pares
        for cropped_name, cropped_ds in cropped_datasets.items():

            names = [ds_name, cropped_name]
            names = [name.replace("pansharpened_", "") for name in names]
            names.sort()
            combined_name = "comb_" + "_".join(names)

            if combined_name not in combinations:
                polygon = box(
                    cropped_ds.x.min(),
                    cropped_ds.y.min(),
                    cropped_ds.x.max(),
                    cropped_ds.y.max(),
                )
                buffered_extent = polygon.buffer(0.005).envelope
                xmin, ymin, xmax, ymax = buffered_extent.bounds

                cropped_main_ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
                cropped_ds = remove_overlapping_pixels(cropped_main_ds, cropped_ds)

                # print(ds_name, cropped_name)
                # print(cropped_main_ds)
                # print(cropped_ds)

                try:
                    result_ds = xr.combine_by_coords(
                        [cropped_main_ds, cropped_ds], combine_attrs="override"
                    )

                    # Store xarray and reload to remove cross-references across objects and reduce memory usage
                    filename = rf"{PROCESSED_DATA_DIR}/tempfiles/{combined_name}.pkl"
                    pickle_xr_dataset(ds, filename)
                    with open(filename, "rb") as f:
                        result_ds = pickle.load(f)
                    to_remove += [filename]

                    combinations[combined_name] = result_ds
                except Exception as e:
                    print(e)

    all_datasets = combinations | datasets

    return all_datasets


def filter_black_pixels_over_dim(ds, dim="x"):
    if dim == "x":
        other_dim = "y"
    elif dim == "y":
        other_dim = "x"
    else:
        raise ValueError("dim must be 'x' or 'y'")

    # Selecciono la mitad de la imagen
    center = int(ds[dim].size / 2)
    edge_data = ds.isel({dim: center})

    # Busco los pixeles con al menos 50 pixeles sin datos
    has_black_pixels = (edge_data["band"] == 0).all(dim="band")
    has_black_pixels_in_row = has_black_pixels.rolling({other_dim: 50}).sum()
    valid_data = (has_black_pixels_in_row == 0) | (has_black_pixels_in_row.isnull())

    # Filtro los datos
    first_valid = valid_data.to_numpy().tolist().index(True)
    last_valid = -valid_data.to_numpy().tolist()[::-1].index(True)

    if first_valid == 0:
        first_valid = None
    if last_valid == 0:
        last_valid = None

    return ds.isel({other_dim: slice(first_valid, last_valid)})


def filter_black_pixels(ds):
    # FIXME: This function is not working properly, it requires Python 3.10-... Should add later,
    #   Technically, now (with NYC images) it is not needed because the images are already cropped.
    return ds 
    y_filtered = filter_black_pixels_over_dim(ds, "y")
    filtered = filter_black_pixels_over_dim(y_filtered, "x")
    return filtered


def generate_matrix_of_files(files):
    """Create a matrix of files to be loaded by xr.open_mfdataset.

    Files are ordered as the original tiles, where R1C3 is the first tile of the third column.
    Run xr.open_mfdataset(matrix, combine="nested", concat_dim=["x", "y"], engine="rasterio") after this.

    Parameters:
    files (list): List of files to be loaded

    Returns:
    matrix (list): List of lists of files to be loaded by xr.open_mfdataset
    """
    files.sort()

    matrix = []
    for col in range(1, 5):
        cols_files = [f for f in files if f"C{col}.tif" in f]
        if len(cols_files) > 0:
            matrix += [cols_files]
    return matrix

def generate_matrix_of_datasets(datasets):
    """Create a matrix of datasets to be merged by xr.combine_nested.

    Files are ordered as the original tiles, where R1C3 is the first tile of the third column.
    Run xr.open_mfdataset(matrix, combine="nested", concat_dim=["x", "y"], engine="rasterio") after this.

    Parameters:
    files (list): List of files to be loaded

    Returns:
    matrix (list): List of lists of files to be loaded by xr.open_mfdataset
    """
    datasets = sorted(datasets, key=lambda element: sorted(element.encoding["source"]))
    print([ds.encoding["source"] for ds in datasets])
    matrix = []
    for row in range(1, 10):
        rows_ds = [ds for ds in datasets if f"_R{row}C" in ds.encoding["source"]]
        if len(rows_ds) > 0:
            matrix += [rows_ds]
    return matrix