import os

import numpy as np
import rasterio as rio
import torch
from dotenv import load_dotenv

os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(os.environ["PROJECT_ROOT"], ".env"))

def get_hs(src):
    vnir = src["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"][()]
    vnir = np.transpose(vnir, [1, 0, 2])
    vnir = vnir / (2 ** 16 - 1)
    vnir = torch.from_numpy(vnir).unsqueeze(0)

    swir = src["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"][()]
    swir = np.transpose(swir, [1, 0, 2])
    swir = swir / (2 ** 16 - 1)
    swir = torch.from_numpy(swir).unsqueeze(0)

    list_cw = list(src.attrs["List_Cw_Swir"])[106:108] + list(src.attrs["List_Cw_Vnir"])[3:]
    list_fwhm = list(src.attrs["List_Fwhm_Swir"])[106:108] + list(src.attrs["List_Fwhm_Vnir"])[3:]

    hs = torch.cat((swir[:, 106:108, :, :], vnir[:, 3:, :, :]), dim=1)

    return hs, list_cw, list_fwhm

def read_band(file_path):
    with rio.open(file_path) as src:
        band = src.read(1).astype(float)/ 10000.0
    return band

def load_bands(band_filenames, band_names, folder_path):
    bands = []
    for band_name in band_names:
        band_file = next((file for file in band_filenames if band_name in file), None)
        if band_file:
            bands.append(read_band(os.path.join(folder_path, band_file)))
    return bands

def get_bounds_sentinel(jp2_path):
    with rio.open(jp2_path) as src:
        bounds = src.bounds  # in native CRS (usually UTM meters)
    return bounds.left, bounds.bottom, bounds.right, bounds.top  # west, south, east, north

def get_epsg_from_jp2(jp2_path):
    with rio.open(jp2_path) as src:
        epsg_code = src.crs.to_epsg()
    return epsg_code

def downsampling(data, factor):
    low = torch.zeros(data.size(0), data.size(1), data.size(2) // factor, data.size(3) // factor)
    for i in range(factor):
        for j in range(factor):
            low += data[:, :, i::factor, j::factor] / factor ** 2
    return low

def classical_pan(ms):
    pan = torch.zeros_like(ms)
    pan = torch.cat([pan, pan[:, :2, :, :]], dim=1)

    canal8 = ms[:, 3, :, :].unsqueeze(1)

    canal48 = ms[:, [2,3], :, :]

    canal48mean = canal48.mean(dim=1, keepdim=True)

    canal8repet = canal8.repeat(1, 3, 1, 1)
    canal48meanrepet = canal48mean.repeat(1, 3, 1, 1)


    pan[:,3:6,:,:] = canal8repet
    pan[:, 0:3, :, :] = canal48meanrepet

    return pan