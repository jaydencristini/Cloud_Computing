import os
import json
import sqlite3  # (kept because you import it; not used below yet)
from datetime import datetime, timedelta, timezone
from pathlib import Path

import earthaccess
import numpy as np
import xarray as xr

# Headless-safe matplotlib for GitHub Actions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # (kept because you import it; not used below yet)


# ----------------------------
# Outputs (Actions-friendly)
# ----------------------------
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

# Keep file sizes small if you ever save figures
from matplotlib import rcParams
rcParams["figure.dpi"] = 80


# ----------------------------
# Auth (non-interactive)
# ----------------------------
auth = earthaccess.login(strategy="environment")
if not auth.authenticated:
    raise RuntimeError(
        "Earthdata login failed. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in GitHub Secrets."
    )

print("earthaccess version:", earthaccess.__version__)


# ----------------------------
# Helpers
# ----------------------------
def _utc_range_last_n_days(days_back: int) -> tuple[str, str]:
    """
    Returns ISO8601 (start, end) in UTC covering the last N days.
    Example: days_back=14 -> start = now-14days at 00:00 UTC, end = now.
    """
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(), now.isoformat()


def _open_merged_dataset(results):
    """
    Your original open_virtual_mfdataset merge pattern, unchanged.
    """
    open_options = {
        "access": "indirect",  # access to cloud data (faster in AWS with "direct")
        "load": True,
        "concat_dim": "time",
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
        "combine_attrs": "override",
    }

    result_root = earthaccess.open_virtual_mfdataset(granules=results, **open_options)
    result_product = earthaccess.open_virtual_mfdataset(granules=results, group="product", **open_options)
    result_geolocation = earthaccess.open_virtual_mfdataset(granules=results, group="geolocation", **open_options)

    return xr.merge([result_root, result_product, result_geolocation])


def _subset_and_mean(ds_merged, lon_bounds, lat_bounds, use_quality_flag: bool = True):
    """
    Subset region and temporal mean (your logic).
    """
    sub = ds_merged.sel(
        longitude=slice(lon_bounds[0], lon_bounds[1]),
        latitude=slice(lat_bounds[0], lat_bounds[1]),
    )

    if use_quality_flag and "main_data_quality_flag" in sub:
        sub = sub.where(sub["main_data_quality_flag"] == 0)

    return sub.mean(dim="time")


def _to_triplets(da: xr.DataArray):
    """
    Convert a 2D lat/lon DataArray to list of [lat, lon, value], filtering NaNs.
    """
    lats = da["latitude"].values
    lons = da["longitude"].values
    data = da.values

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    triplets = np.column_stack((lat_grid.flatten(), lon_grid.flatten(), data.flatten()))
    triplets = triplets[~np.isnan(triplets[:, 2])]

    return triplets.tolist()


def _write_json(name: str, data, also_write_public: bool):
    """
    Writes to outputs/<name>.json always; optionally also public/<name>.json
    """
    out_path = OUTPUT_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"Wrote: {out_path}")

    if also_write_public:
        pub_path = PUBLIC_DIR / f"{name}.json"
        with open(pub_path, "w") as f:
            json.dump(data, f)
        print(f"Wrote: {pub_path}")


# ----------------------------
# Your fetch functions
# (same shape, but time window is dynamic)
# ----------------------------
def No2Fetch(days_back: int = 14, count: int = 14):
    start_iso, end_iso = _utc_range_last_n_days(days_back)

    results = earthaccess.search_data(
        short_name="TEMPO_NO2_L3",
        version="V03",
        temporal=(start_iso, end_iso),
        count=count,
    )

    print(f"NO2 granules found: {len(results)}")
    if not results:
        return []

    result_merged = _open_merged_dataset(results)

    lon_bounds = (-137, -50)
    lat_bounds = (17, 56)

    print(
        f"Analyzing region: {lat_bounds[0]}°N to {lat_bounds[1]}°N, "
        f"{abs(lon_bounds[0])}°W to {abs(lon_bounds[1])}°W"
    )

    temporal_mean_ds = _subset_and_mean(result_merged, lon_bounds, lat_bounds, use_quality_flag=True)
    print(f"Dataset shape after subsetting: {temporal_mean_ds.dims}")

    mean_vertical_column_trop = temporal_mean_ds["vertical_column_troposphere"].compute()

    triplet_list = _to_triplets(mean_vertical_column_trop)
    print(f"Extracted {len(triplet_list)} data points")

    magnitude = 0.2 * (10 ** 16)  # 2e15

    filtered_triplets = []
    max_value = 0.0

    for t in triplet_list:
        if t[2] >= magnitude:
            filtered_triplets.append(t)
        if t[2] > max_value:
            max_value = t[2]

    if max_value > 0:
        for item in filtered_triplets:
            item[2] = 0.5 + (item[2] / max_value * 0.5)

    print(f"Kept {len(filtered_triplets)} of {len(triplet_list)} data points")
    return filtered_triplets


def FromaldehydeFetch(days_back: int = 14, count: int = 14):
    start_iso, end_iso = _utc_range_last_n_days(days_back)

    results = earthaccess.search_data(
        short_name="TEMPO_HCHO_L3",
        version="V03",
        temporal=(start_iso, end_iso),
        count=count,
    )

    print(f"HCHO granules found: {len(results)}")
    if not results:
        return []

    result_merged = _open_merged_dataset(results)

    lon_bounds = (-137, -50)
    lat_bounds = (17, 56)

    print(
        f"Analyzing region: {lat_bounds[0]}°N to {lat_bounds[1]}°N, "
        f"{abs(lon_bounds[0])}°W to {abs(lon_bounds[1])}°W"
    )

    temporal_mean_ds = _subset_and_mean(result_merged, lon_bounds, lat_bounds, use_quality_flag=True)
    print(f"Dataset shape after subsetting: {temporal_mean_ds.dims}")

    mean_vertical_column_trop = temporal_mean_ds["vertical_column"].compute()

    triplet_list = _to_triplets(mean_vertical_column_trop)
    print(f"Extracted {len(triplet_list)} data points")

    max_value = 0.0
    for t in triplet_list:
        if t[2] > max_value:
            max_value = t[2]

    filtered_triplets = []
    threshold = max_value * 0.1

    for t in triplet_list:
        if t[2] > threshold:
            filtered_triplets.append(t)

    if max_value > 0:
        for item in filtered_triplets:
            item[2] = 0.5 + (item[2] / max_value * 0.5)

    print(f"Kept {len(filtered_triplets)} of {len(triplet_list)} data points")
    return filtered_triplets


def OzoneFetch(days_back: int = 14, count: int = 14):
    start_iso, end_iso = _utc_range_last_n_days(days_back)

    results = earthaccess.search_data(
        short_name="TEMPO_O3TOT_L3",
        version="V03",
        temporal=(start_iso, end_iso),
        count=count,
    )

    print(f"O3 granules found: {len(results)}")
    if not results:
        return []

    result_merged = _open_merged_dataset(results)

    lon_bounds = (-137, -50)
    lat_bounds = (17, 56)

    print(
        f"Analyzing region: {lat_bounds[0]}°N to {lat_bounds[1]}°N, "
        f"{abs(lon_bounds[0])}°W to {abs(lon_bounds[1])}°W"
    )

    temporal_mean_ds = _subset_and_mean(result_merged, lon_bounds, lat_bounds, use_quality_flag=False)
    print(f"Dataset shape after subsetting: {temporal_mean_ds.dims}")

    mean_vertical_column_trop = temporal_mean_ds["o3_below_cloud"].compute()

    triplet_list = _to_triplets(mean_vertical_column_trop)
    print(f"Extracted {len(triplet_list)} data points")

    filtered_triplets = []
    max_value = 0.0

    for t in triplet_list:
        if 10 < t[2] < 100:
            filtered_triplets.append(t)
            if t[2] > max_value:
                max_value = t[2]

    if max_value > 0:
        for item in filtered_triplets:
            item[2] = 0.5 + (item[2] / max_value * 0.5)

    print(f"Kept {len(filtered_triplets)} of {len(triplet_list)} data points")
    return filtered_triplets


def AerosolFetch(days_back: int = 14, count: int = 14):
    """
    NOTE: Your original code searched TEMPO_O3TOT_L3 but then read uv_aerosol_index.
    I am keeping that behavior exactly to match your current framework.
    If this ever fails with "variable not found", we can swap the short_name later.
    """
    start_iso, end_iso = _utc_range_last_n_days(days_back)

    results = earthaccess.search_data(
        short_name="TEMPO_O3TOT_L3",
        version="V03",
        temporal=(start_iso, end_iso),
        count=count,
    )

    print(f"Aerosol (via O3TOT search) granules found: {len(results)}")
    if not results:
        return []

    result_merged = _open_merged_dataset(results)

    lon_bounds = (-137, -50)
    lat_bounds = (17, 56)

    print(
        f"Analyzing region: {lat_bounds[0]}°N to {lat_bounds[1]}°N, "
        f"{abs(lon_bounds[0])}°W to {abs(lon_bounds[1])}°W"
    )

    temporal_mean_ds = _subset_and_mean(result_merged, lon_bounds, lat_bounds, use_quality_flag=False)
    print(f"Dataset shape after subsetting: {temporal_mean_ds.dims}")

    mean_vertical_column_trop = temporal_mean_ds["uv_aerosol_index"].compute()

    triplet_list = _to_triplets(mean_vertical_column_trop)
    print(f"Extracted {len(triplet_list)} data points")

    filtered_triplets = []
    max_value = 0.0
    min_value = 0.0

    for t in triplet_list:
        if -1 < t[2] < 30:
            filtered_triplets.append(t)
            if t[2] > max_value:
                max_value = t[2]
            elif t[2] < min_value:
                min_value = t[2]

    denom = (max_value - min_value)
    if denom != 0:
        for item in filtered_triplets:
            item[2] = 0.5 + ((item[2] - min_value) / denom) * 0.5

    print(f"Kept {len(filtered_triplets)} of {len(triplet_list)} data points")
    return filtered_triplets


# ----------------------------
# Main (choose what to run)
# ----------------------------
if __name__ == "__main__":
    # You can control these via environment variables in GitHub Actions if you want.
    DAYS_BACK = int(os.environ.get("DAYS_BACK", "14"))
    COUNT = int(os.environ.get("COUNT", "14"))
    MODE = os.environ.get("MODE", "aerosol")  # aerosol | no2 | hcho | o3
    WRITE_PUBLIC = os.environ.get("WRITE_PUBLIC", "0") == "1"

    if MODE == "no2":
        data = No2Fetch(days_back=DAYS_BACK, count=COUNT)
        _write_json("no2", data, also_write_public=WRITE_PUBLIC)

    elif MODE == "hcho":
        data = FromaldehydeFetch(days_back=DAYS_BACK, count=COUNT)
        _write_json("hcho", data, also_write_public=WRITE_PUBLIC)

    elif MODE == "o3":
        data = OzoneFetch(days_back=DAYS_BACK, count=COUNT)
        _write_json("o3", data, also_write_public=WRITE_PUBLIC)

    else:
        # default: aerosol
        data = AerosolFetch(days_back=DAYS_BACK, count=COUNT)
        _write_json("aerosol", data, also_write_public=WRITE_PUBLIC)
