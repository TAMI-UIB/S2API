import glob
import os

import h5py
import numpy as np
import rasterio as rio
import torch
from torch.nn.functional import interpolate

hdf5_dir = "hdf5"
gtif_dir = "gtif"
if __name__ == '__main__':

    for file in glob.iglob(os.path.join(hdf5_dir, "*.he5")):
        print(file)
        with h5py.File(file) as src:
            epsg = src.attrs["Epsg_Code"]
            west = min(src.attrs["Product_LLcorner_easting"], src.attrs["Product_ULcorner_easting"])
            south = min(src.attrs["Product_LLcorner_northing"], src.attrs["Product_LRcorner_northing"])
            east = max(src.attrs["Product_LRcorner_easting"], src.attrs["Product_URcorner_easting"])
            north = max(src.attrs["Product_ULcorner_northing"], src.attrs["Product_URcorner_northing"])

            vnir = src["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"][()]
            swir = src["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"][()]
            pan = src['HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube'][()]
        data = vnir[:, 3:, :]
        data = data / (2.**16-1.)
        data = data
        data = np.transpose(data, [1, 0, 2])
        pan = np.expand_dims(pan, axis=0) / (2.**16-1.)

        data = torch.from_numpy(data).unsqueeze(0)
        pan = torch.from_numpy(pan).unsqueeze(0)
        data = data.type(torch.float16)
        pan = pan.type(torch.float16)
        data = interpolate(data, scale_factor=6)
        data = torch.cat((data, pan), dim=1)
        data = 255 * data
        data = data[0]
        data = data.numpy().astype(np.uint8)
        count, height, width = data.shape

        profile = {
            "fp": file.replace(hdf5_dir, gtif_dir).replace(".he5", ".tif"),
            "mode": "w",
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": count,
            "crs": rio.crs.CRS.from_epsg(epsg),
            "transform": rio.transform.from_bounds(west, south, east, north, width, height),
            "dtype": data.dtype,
        }

        with rio.open(**profile) as dst:
            for i in range(count):
                dst.write(data[i], i + 1)