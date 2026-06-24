import threading
import pystac_client
import planetary_computer
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import numpy as np

_local = threading.local()

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
)

def get_catalog() -> pystac_client.Client:
    """One STAC client per thread — avoids connection pool contention."""
    if not hasattr(_local, "catalog"):
        _local.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    return _local.catalog

def fetch_naip_crop(
    lon: float, lat: float,
    crop_size_meters: float,
    nbands: int = 4,
    out_pixels: int = 250,
    year_hint: int | None = None,
) -> tuple[np.ndarray | None, int | None]:
    """
    Fetch a single NAIP crop from Planetary Computer.
    
    Returns:
        (crop, actual_year) where crop is (C, H, W) uint8 or None,
        and actual_year is the year of the NAIP image used.
    """
    # 1. Convert (lon, lat) center + crop_size_meters to EPSG:4326 bbox
    # Rough approximation: 1 degree latitude = ~111km
    # longitude degrees depend on latitude
    lat_deg = (crop_size_meters / 2) / 111_000
    lon_deg = (crop_size_meters / 2) / (111_000 * np.cos(np.radians(lat)))
    
    bbox = [lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg]

    try:
        items = list(get_catalog().search(
            collections=["naip"],
            bbox=bbox,
            max_items=50,
        ).items())
    except Exception as e:
        return None, None

    if not items:
        return None, None

    # Sort items by proximity to year_hint if provided
    if year_hint is not None:
        items.sort(key=lambda item: abs(item.datetime.year - year_hint))

    item = items[0]
    actual_year = item.datetime.year
    
    asset = item.assets.get("image") or item.assets.get("visual")
    if not asset:
        return None, None

    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(asset.href) as src:
                native_bb = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
                window = from_bounds(*native_bb, transform=src.transform)
                
                # Request out_shape with out_pixels size
                # And only the required number of bands
                channels = min(nbands, src.count)
                crop = src.read(
                    indexes=list(range(1, channels + 1)),
                    window=window,
                    out_shape=(channels, out_pixels, out_pixels),
                    resampling=Resampling.bilinear,
                )
                
                # If requested more bands than available, pad with zeros
                if channels < nbands:
                    padded_crop = np.zeros((nbands, out_pixels, out_pixels), dtype=crop.dtype)
                    padded_crop[:channels] = crop
                    crop = padded_crop
                    
                return crop, actual_year
    except Exception as e:
        return None, None
