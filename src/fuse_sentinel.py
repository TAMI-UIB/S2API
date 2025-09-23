import argparse
import glob
import os

import h5py
import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.coords import BoundingBox
import torch
from dotenv import load_dotenv
from torchvision.utils import save_image
from tqdm import tqdm

from src.base import Experiment
from src.utils.enhancement import prisma_correction
from src.utils.index import IndexCalculator
from src.utils.patchwork import CreatePatches

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

def test(ckpt, device, nickname, place, dataset_path):
    cfg = ckpt['cfg']
    cfg.devices = [0]

    weights = ckpt['state_dict']
    experiment = Experiment(cfg)
    experiment.load_state_dict(weights)
    model = experiment.model.to(device)
    model.eval()

    R10_bands = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2']
    R20_bands = ['B05_20m.jp2', 'B06_20m.jp2', 'B07_20m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']

    for img_name in os.listdir(dataset_path):

        r10m_folder = os.path.join(dataset_path, img_name, 'R10m')
        r20m_folder = os.path.join(dataset_path, img_name, 'R20m')

        # List files in the folders
        R10files = os.listdir(r10m_folder)
        R20files = os.listdir(r20m_folder)

        # Load R10 bands
        r10bands = load_bands(R10files, R10_bands, r10m_folder)
        R10img = torch.stack([torch.from_numpy(band).unsqueeze(0) for band in r10bands], dim=1)

        # Load R20 bands
        r20bands = load_bands(R20files, R20_bands, r20m_folder)
        R20img = torch.stack([torch.from_numpy(band).unsqueeze(0) for band in r20bands], dim=1)

        # Define coordinates
        info_path = os.path.join(r10m_folder, R10files[1])
        west, south, east, north = get_bounds_sentinel(info_path)
        epsg = get_epsg_from_jp2(info_path)

        # Set
        hs = R20img
        ms = R10img
        pan = classical_pan(ms)

        hs = hs.to(torch.float32)
        pan = pan.to(torch.float32)
        ms = ms.to(torch.float32)

        N, C, h, w = hs.size()
        _, _, H, W = ms.size()

        patch_size = 64
        h_rm = h % patch_size
        h_new = h - h_rm
        H_new = H - h_rm * 2
        w_rm = w % patch_size
        w_new = w - w_rm
        W_new = W - w_rm * 2
        new_south = south + (north - south) * (h_rm / h)
        new_east = east - (east - west) * (w_rm / w)
        hs = hs[:, :, :h_new, :w_new]
        pan = pan[:, :, :H_new, :W_new]
        ms = ms[:, :, :H_new, :W_new]


        save_image(ms[0, [2,1,0], :, :], f'/home/dani/results/MaLiSatDetection/Sentinel2imgs/ms_rgb{nickname}_{img_name}.png')

        N, _, H, W = pan.size()
        _, C, _, _ = hs.size()

        hs_patcher = CreatePatches(hs, patch_size, False)
        hs_patches = hs_patcher.do_patches(hs)
        pan_patcher = CreatePatches(pan, patch_size * 2, False)
        pan_patches = pan_patcher.do_patches(pan)
        fused = []
        # for using tqdm

        for i in tqdm(range(hs_patches.size(0))):
            hs_p = hs_patches[[i]]
            pan_p = pan_patches[[i]]
            with torch.no_grad():
                hs_p = hs_p.to(device)
                pan_p = pan_p.to(device)
                fused_p = model(hs=hs_p, pan=pan_p)
                fused_p = fused_p['pred']
                fused.append(fused_p.cpu())

        fused = torch.cat(fused, dim=0)

        pan_patcher.C = hs_patcher.C
        fused = pan_patcher.undo_patches(fused)


        data = torch.cat((fused, ms), dim=1).squeeze()
        data = ((2**8-1) * data.numpy()).astype(np.uint8)

        count, height, width = data.shape

        file= f"/home/dani/datasets/Sentinel2/{img_name}_fused.tif"


        ### FINS AQUÍ!

        profile = {
            "fp": file,
            "mode": "w",
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": count,
            "crs": rio.crs.CRS.from_epsg(epsg),
            "transform": rio.transform.from_bounds(west, new_south, new_east, north, width, height),
            "dtype": data.dtype,
        }

        with rio.open(**profile) as dst:
            for i in range(count):
                dst.write(data[i], i + 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fuse script")
    parser.add_argument('--log_path', type=str, required=True, help='Path of the log file')
    parser.add_argument('--device', type=str, default='cuda:1', help='cuda:0')
    parser.add_argument('--nickname', type=str, default='All')
    parser.add_argument('--place', type=str, default='all')
    parser.add_argument('--dataset_path', type=str, default='/home/dani/datasets/Sentinel2')

    args = parser.parse_args()
    device = args.device
    ckpt = torch.load(f"{args.log_path}/checkpoints/best.ckpt", map_location=args.device, weights_only=False)
    test(ckpt, args.device, args.nickname, args.place, args.dataset_path)