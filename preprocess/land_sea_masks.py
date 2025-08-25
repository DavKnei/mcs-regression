import xarray as xr
import numpy as np
from matplotlib.path import Path
from pathlib import (
    Path as PPath,
)  # Use PPath to avoid conflict with matplotlib.path.Path


def create_polygon_mask(lsm_data, polygon_points):
    """
    Creates a sea mask from a list of polygon points.

    Args:
        lsm_data (xr.DataArray): The base land-sea mask data.
        polygon_points (list): A list of (longitude, latitude) tuples defining the polygon.

    Returns:
        xr.DataArray: A new mask with sea points (0) inside the polygon and land (1) elsewhere.
    """
    # Create a matplotlib Path object from the polygon points
    polygon_path = Path(polygon_points)

    # Create a grid of all longitude and latitude points from the data
    lon_grid, lat_grid = np.meshgrid(lsm_data.longitude, lsm_data.latitude)
    points = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T

    # Find all grid points inside the polygon
    grid_points_inside_polygon = polygon_path.contains_points(points)
    inside_polygon_mask = grid_points_inside_polygon.reshape(lsm_data.shape)

    # Start with a DataArray of all land (value = 1)
    new_mask = xr.full_like(lsm_data, fill_value=1, dtype=np.float32)

    # Set points to sea (0) if they are inside the polygon AND are sea in the original data
    new_mask = new_mask.where(~((inside_polygon_mask) & (lsm_data < 0.5)), 0)
    return new_mask


# --- 1. Configuration ---

# !!! IMPORTANT: Update this path to your land-sea mask file !!!
ERA5_LAND_SEA_MASK_PATH = (
    "/reloclim/dkn/data/regression_data/mask/era5_land_sea_mask.nc"
)
OUTPUT_DIR = PPath("./data/")
OUTPUT_FILENAME = OUTPUT_DIR / "moisture_source_masks.nc"

# Define all polygon points for each moisture source
MASKS_CONFIG = {
    "Tyrrhenian_Sea": {
        "long_name": "Tyrrhenian Sea Mask",
        "points": [
            (17, 39),
            (14.5, 37.3),
            (8.5, 39),
            (9, 41.5),
            (9, 42.7),
            (13, 43),
        ],
    },
    "Ligurian_Sea": {
        "long_name": "Ligurian Sea Mask",
        "points": [(7.5, 43.8), (8.0, 42.8), (11, 42.5), (11, 44.1), (9, 46)],
    },
    "Adriatic_Sea": {
        "long_name": "Adriatic Sea Mask",
        "points": [(18.5, 40.0), (21, 40.2), (18.0, 43.5), (12.0, 47), (12.0, 42)],
    },
    "Ionian_Sea": {
        "long_name": "Ionian Sea Mask",
        "points": [
            (16, 41),
            (20.0, 39.5),
            (21.0, 38.0),
            (22.5, 36.5),
            (14, 36.8),
            (16.0, 38.5),
        ],
    },
    "Balearic_Sea": {
        "long_name": "Balearic Sea Mask",
        "points": [
            (3.0, 45),
            (-1, 39.5),
            (0.5, 38.6),
            (8.7, 38.9),
            (9.2, 41.3),
            (8.5, 42.5),
            (6.7, 43.5),
        ],
    },
    "Black_Sea": {
        "long_name": "Black Sea Mask",
        "points": [(28, 41), (42, 41), (42, 47), (28, 47)],
    },
    "Atlantic_France": {
        "long_name": "Atlantic Ocean Mask west to France",
        "points": [(-10, 43), (-10, 49), (-1, 49), (-1, 43)],
    },
    "Atlantic_Spain": {
        "long_name": "Atlantic Ocean Mask west to Spain",
        "points": [(-15, 35), (-15, 43), (-6, 43), (-6, 35)],
    },
}

# --- 2. Main Execution ---

if __name__ == "__main__":
    print("Starting mask generation process...")

    # Load and prepare the base land-sea mask
    try:
        ds_lsm = xr.open_dataset(ERA5_LAND_SEA_MASK_PATH)
        # Squeeze unnecessary time dimensions if they exist
        if "valid_time" in ds_lsm.dims:
            ds_lsm = ds_lsm.squeeze(dim="valid_time", drop=True)
        if "time" in ds_lsm.dims:
            ds_lsm = ds_lsm.squeeze(dim="time", drop=True)
        lsm_data = ds_lsm["lsm"]
        print(f"Successfully loaded land-sea mask from: {ERA5_LAND_SEA_MASK_PATH}")

        # --- FIX IS HERE ---
        # Convert longitude from 0-360 range to -180 to 180 range to match polygons
        print("Standardizing longitude coordinates to -180 to 180 range...")
        lsm_data = lsm_data.assign_coords(
            longitude=(((lsm_data.longitude + 180) % 360) - 180)
        )
        lsm_data = lsm_data.sortby("longitude")
        # --- END OF FIX ---

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{ERA5_LAND_SEA_MASK_PATH}'.")
        print("Please update the ERA5_LAND_SEA_MASK_PATH variable in the script.")
        exit()

    # A list to hold all the created mask DataArrays
    all_masks = []

    # Loop through the configuration to create each mask
    for var_name, config in MASKS_CONFIG.items():
        print(f"  - Creating mask for: {config['long_name']}...")

        mask_da = create_polygon_mask(lsm_data, config["points"])
        mask_da.name = var_name
        mask_da.attrs["long_name"] = config["long_name"]
        mask_da.attrs[
            "source"
        ] = "Generated via polygon method from ERA5 land-sea mask."

        all_masks.append(mask_da)

    # Merge all DataArrays into a single Dataset
    final_dataset = xr.merge(all_masks)

    # Add global attributes to the final file
    final_dataset.attrs["title"] = "Moisture Source Masks for MCS analysis over Europe"
    final_dataset.attrs[
        "description"
    ] = "Contains multiple sea masks as separate data variables."

    # Check if 'number' and 'expver' exist before dropping
    vars_to_drop = [var for var in ["number", "expver"] if var in final_dataset.coords]
    if vars_to_drop:
        final_dataset = final_dataset.drop_vars(vars_to_drop)

    lon_slice = slice(-20, 40)
    lat_slice = slice(
        70, 30
    )  # Use (max, min) for latitude because it's often descending

    # Apply the slice to the dataset
    final_dataset = final_dataset.sel(longitude=lon_slice, latitude=lat_slice)

    # Save the dataset to a single NetCDF file
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        final_dataset.to_netcdf(OUTPUT_FILENAME, engine="netcdf4")
        print(f"\n✅ All masks successfully created and saved to:")
        print(f"   {OUTPUT_FILENAME.resolve()}")
    except Exception as e:
        print(f"\n❌ An error occurred while saving the file: {e}")
