import rasterio as rio
import rasterio.transform as transform
from rasterio.profiles import DefaultGTiffProfile
import numpy as np
import pandas as pd
import glob
import warnings

def ArrayToRaster(arr, filename, sample_raster, mask):
    # Ignore the warning for not having a georeference
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # Load sample raster metadata
    with rio.open(sample_raster, 'r') as src:
        meta = src.meta.copy()
        if mask is None:
            mask = src.read_masks(1).astype(bool)

    # Update metadata for output raster
    meta.update({
        'count': 1,
        'dtype': arr.dtype,
        'height': arr.shape[0],
        'width': arr.shape[1],
        'nodata': -9999,  # set nodata value here
        # 'transform': transform.Affine(1, 0, meta['transform'][2],         #use this to create raster
        #                               0, 1, meta['transform'][5])
    })

    # Create output raster file
    with rio.open(filename, 'w', **meta) as dst:
        # Mask array and write to raster file
        arr_masked = np.where(mask, arr, meta['nodata'])
        dst.write(arr_masked, 1)

def Velocity_raster(arr, filename, sample_raster, mask=None):
    # Ignore the warning for not having a georeference
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # Load sample raster metadata
    with rio.open(sample_raster, 'r') as src:
        meta = src.meta.copy()
        if mask is None:
            mask = src.read_masks(1).astype(bool)

    # Update metadata for output raster
    meta.update({
        'count': arr.shape[0],
        'dtype': arr.dtype,
        'height': arr.shape[1],
        'width': arr.shape[2],
        'nodata': -9999,  # set nodata value here
        # 'transform': transform.Affine(1, 0, meta['transform'][2],         #use this to create raster
        #                               0, 1, meta['transform'][5])
    })

    # Create output raster file
    with rio.open(filename, 'w', **meta) as dst:
        for i in range(arr.shape[0]):
            # Mask array and write to raster file
            arr_masked = np.where(mask, arr[i], meta['nodata'])
            dst.write(arr_masked, i+1)
        
def VelocitytoRasterIO(velx, vely, dst_filename, sample_raster, mask=None):
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)
    with rio.open(sample_raster, 'r') as src:
        naip_meta = src.meta.copy()

    naip_meta['count'] = 2
    naip_meta['nodata'] = -9999
    naip_meta['dtype'] = 'float32'

    # write your the ndvi raster object
    with rio.open(dst_filename, 'w', **naip_meta) as dst:
        velx_masked = np.where(mask, velx, naip_meta['nodata'])
        vely_masked = np.where(mask, vely, naip_meta['nodata'])
        dst.write(velx_masked, 1)
        dst.write(vely_masked, 2)

def RasterToArray(dem_file):
    # Ignore the warning for not having a georeference
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # Open the DEM file
    with rio.open(dem_file) as src:
        DEM = src.read(1)
        mask = src.read_masks(1)
        bounds = [src.bounds.left, src.bounds.top,
                  src.bounds.right, src.bounds.bottom]
        geotransform = src.transform
        cell_size = geotransform[0]

    DEM = DEM.astype(np.double)
    bounds = np.array(bounds)
    mask = ~mask.astype(bool)

    # Create a wall all around the domain
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    return DEM, mask, bounds, cell_size

def DEMGenerate(npa, dst_filename, mask=None):
    """this function is used to generate digital elevation model (DEM file) of
    the 1st layer (band = 1) using a numpy array"""
    profile = DefaultGTiffProfile(count=1)
    profile['nodata'] = -999
    profile['width'] = npa.shape[0]
    profile['height'] = npa.shape[1]
    profile['dtype'] = 'float32'
    profile['blockxsize'] = 128
    profile['blockysize'] = 128
    profile['transform'] = transform.Affine(1, 0, 0, 0, 1, -npa.shape[0])

    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    if mask is not None:
        npa = np.ma.masked_array(npa, mask)

    with rio.open(dst_filename, 'w', **profile) as dst:
        dst.write(npa, 1)


def InverseWeightedDistance(x, y, v, grid, power):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distance = np.sqrt((x-i)**2+(y-j)**2)
            if (distance**power).min() == 0:
                grid[i, j] = v[(distance**power).argmin()]
            else:
                total = np.sum(1/(distance**power))
                grid[i, j] = np.sum(v/(distance**power)/total)
    return grid

def merge_csv(directory_path):
    
    csv_files = glob.glob(directory_path + "/*.csv")
    csv_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    dataframes = []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        dataframes.append(df)
        if i != len(csv_files) - 1:
            gap = pd.DataFrame([None] * 1)
            dataframes.append(gap)
        
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv('merged_file.csv', index=False)

import numpy as np

# def InverseWeightedDistance(data, grid, power, x_col=0, y_col=1, v_col=2):
#     for i in range(grid.shape[0]):
#         for j in range(grid.shape[1]):
#             distances = np.sqrt((data[:, x_col] - i)**2 + (data[:, y_col] - j)**2)
#             if (distances**power).min() == 0:
#                 grid[i, j] = data[(distances**power).argmin(), v_col]
#             else:
#                 weights = 1 / (distances**power)
#                 total = np.sum(weights)
#                 grid[i, j] = np.sum(data[:, v_col] * weights / total)
#     return grid

