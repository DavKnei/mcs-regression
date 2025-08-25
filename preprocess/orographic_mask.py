import xarray as xr
import numpy as np
from matplotlib.path import Path
from pathlib import Path as PPath

def create_orography_mask(geopotential_data, polygon_points, height_threshold_m, G=9.80665):
    """
    Creates a mountain mask from a geopotential dataset based on a polygon and height threshold.

    Args:
        geopotential_data (xr.DataArray): The base geopotential data ('z' variable).
        polygon_points (list): A list of (longitude, latitude) tuples defining the polygon.
        height_threshold_m (float): The minimum elevation in meters to be included in the mask.
        G (float): Gravitational constant.

    Returns:
        xr.DataArray: A new mask with high-elevation points (1) inside the polygon and low (0) elsewhere.
    """
    # Convert geopotential to height in meters
    height_data = geopotential_data / G

    # Create a matplotlib Path object from the polygon points
    polygon_path = Path(polygon_points)

    # Create a grid of all longitude and latitude points
    lon_grid, lat_grid = np.meshgrid(height_data.lon, height_data.lat)
    points = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T

    # Find all grid points inside the polygon
    grid_points_inside_polygon = polygon_path.contains_points(points)
    inside_polygon_mask = grid_points_inside_polygon.reshape(height_data.shape)

    # Start with a DataArray of all zeros (no mountains)
    new_mask = xr.full_like(height_data, fill_value=0, dtype=np.float32)

    # Create the final boolean condition: inside the polygon AND above the height threshold
    final_condition = (inside_polygon_mask) & (height_data >= height_threshold_m)

    # Where the condition is True, set the mask value to 1
    new_mask = new_mask.where(~final_condition, 1)

    return new_mask

# --- 1. Configuration ---

# Paths and constants
ERA5_GEOPOTENTIAL_PATH = "/reloclim/dkn/data/regression_data/mask/surface_geopot.nc"
OUTPUT_DIR = PPath("./data/")
OUTPUT_FILENAME = OUTPUT_DIR / "orography_masks.nc"
G = 9.80665  # m/s^2

# Define polygons for major European mountain ranges with individual height thresholds
OROGRAPHY_CONFIG = {
    'Alps': {
        'long_name': 'Alps Orography Mask',
        'points': [
            (4.5, 43), (8, 43), (8, 45), (17, 46), (17, 48), (10, 48), (5, 47)
        ],
        'height_threshold': 800  # Example of a custom threshold for the Alps
    },
    'Pyrenees': {
        'long_name': 'Pyrenees Orography Mask',
        'points': [
            (-1.5, 42), (3, 41.5), (3, 43), (-1.5, 43.5)
        ],
        'height_threshold': 800 # Default threshold
    },
    'Carpathians': {
        'long_name': 'Carpathian Mountains Orography Mask',
        'points': [
            (21, 44), (27, 45), (27, 47), (24, 49.5), (22, 48.5)
        ],
        'height_threshold': 600 # Example of a lower threshold for a different range
    },
    'Dinaric_Alps': {
        'long_name': 'Dinaric Alps Orography Mask',
        'points': [
            (14.5, 46), (22, 43.5), (22, 41), (18.5, 42.5), (15, 42)
        ],
        'height_threshold': 500 # Default threshold
    }
}

# --- 2. Main Execution ---

if __name__ == "__main__":
    print("Starting orography mask generation process...")

    # Load and prepare the base geopotential data
    try:
        ds_geo = xr.open_dataset(ERA5_GEOPOTENTIAL_PATH)
        geopotential_data = ds_geo['Z'].squeeze(drop=True)
        print(f"Successfully loaded geopotential data from: {ERA5_GEOPOTENTIAL_PATH}")

        # Standardize longitude coordinates to -180 to 180 range
        print("Standardizing longitude coordinates...")
        geopotential_data = geopotential_data.assign_coords(lon=(((geopotential_data.lon + 180) % 360) - 180))
        geopotential_data = geopotential_data.sortby('lon')

    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: Could not load data. Check path and variable name ('z'). Details: {e}")
        exit()

    all_masks = []

    # Create each mask
    for var_name, config in OROGRAPHY_CONFIG.items():
        # --- MODIFIED SECTION ---
        # Get the specific height threshold for this range, or use 1000 as a default
        height_thresh = config.get('height_threshold', 1000)
        
        print(f"  - Creating mask for: {config['long_name']} (Threshold: {height_thresh}m)...")
        
        # Pass the specific threshold to the function
        mask_da = create_orography_mask(geopotential_data, config['points'], height_thresh, G)
        
        mask_da.name = var_name
        mask_da.attrs['long_name'] = config['long_name']
        # Store the specific threshold used in the metadata
        mask_da.attrs['height_threshold_m'] = height_thresh
        all_masks.append(mask_da)
        # --- END OF MODIFIED SECTION ---

    # Merge into a single Dataset
    final_dataset = xr.merge(all_masks)
    final_dataset.attrs['title'] = 'Orography Masks for Major European Mountain Ranges'
    final_dataset.attrs['description'] = 'Masks identify terrain above individually specified height thresholds within defined polygons.'

    lon_slice = slice(-20, 40)
    lat_slice = slice(
        70, 30
    )  # Use (max, min) for latitude because it's often descending

    # Apply the slice to the dataset
    final_dataset = final_dataset.sel(lon=lon_slice, lat=lat_slice)

    # Save to NetCDF file
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        final_dataset.to_netcdf(OUTPUT_FILENAME, engine="netcdf4")
        print(f"\n✅ All orography masks successfully created and saved to:")
        print(f"   {OUTPUT_FILENAME.resolve()}")
    except Exception as e:
        print(f"\n❌ An error occurred while saving the file: {e}")