import argparse
import os
import rasterio as rio
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from src.base import Experiment
from src.utils.patchwork import CreatePatches

from src.app_scripts.downloader import BandDownloader
from src.app_scripts.band_loader_functions import *

import torch.nn.functional as F

def bicubic_interpolate(img, scale_factor=None, size=None):
    return F.interpolate(img, size=size, scale_factor=scale_factor, mode='bicubic', align_corners=False)

def test(ckpt, device, nickname, dataset_path):
    cfg = ckpt['cfg']
    cfg.devices = [0]

    print("Loading model...")

    weights = ckpt['state_dict']
    experiment = Experiment(cfg)
    experiment.load_state_dict(weights)
    model = experiment.model.to(device)
    model.eval()

    print('Model loaded')

    R10_bands = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2']
    R20_bands = ['B05_20m.jp2', 'B06_20m.jp2', 'B07_20m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']
    R60_bands = ['B01_60m.jp2', 'B09_60m.jp2']

    for img_name in os.listdir(dataset_path):

        r10m_folder = os.path.join(dataset_path, img_name, 'R10m')
        r20m_folder = os.path.join(dataset_path, img_name, 'R20m')
        r60m_folder = os.path.join(dataset_path, img_name, 'R60m')

        # List files in the folders
        R10files = os.listdir(r10m_folder)
        R20files = os.listdir(r20m_folder)
        R60files = os.listdir(r60m_folder)

        # Load R10 bands
        r10bands = load_bands(R10files, R10_bands, r10m_folder)
        R10img = torch.stack([torch.from_numpy(band).unsqueeze(0) for band in r10bands], dim=1)

        # Load R20 bands
        r20bands = load_bands(R20files, R20_bands, r20m_folder)
        R20img = torch.stack([torch.from_numpy(band).unsqueeze(0) for band in r20bands], dim=1)

        # Load R60 bands
        r60bands = load_bands(R60files, R60_bands, r60m_folder)
        R60img = torch.stack([torch.from_numpy(band).unsqueeze(0) for band in r60bands], dim=1)

        # Define coordinates
        info_path = os.path.join(r10m_folder, R10files[1])
        west, south, east, north = get_bounds_sentinel(info_path)
        epsg = get_epsg_from_jp2(info_path)

        print('Bands read')

        # Set
        hs = R20img
        #print(f'HS size: {hs.size()}')
        ms = R10img
        kks = R60img
        
        results_dir = f'{dataset_path}/{img_name}/Resultats'
        os.makedirs(results_dir, exist_ok=True)

        save_image(ms[0, [2, 1, 0], :, :], f"{results_dir}/Abans_abans_de_res_{img_name}.png")

        print(f'MS size: {ms.size()}')
        print(f'HS size: {hs.size()}')
        print(f'KKs size: {kks.size()}')
        # pan = classical_pan(ms)
        # print(f'PAN size: {pan.size()}')

        hs = hs[:, :, :500, :500].to(torch.float32)#[:, :, :500, :500]
        # pan = pan.to(torch.float32)
        ms = ms[:, :, :1000, :1000].to(torch.float32)#[:, :, :1000, :1000]
        kks = kks[:, :, :200, :200].to(torch.float32)#[:, :, :200, :200]

        last_folder = dataset_path.split("/")[-1]
        results_dir = f'{dataset_path}/{img_name}/Resultats'
        os.makedirs(results_dir, exist_ok=True)
        save_image(ms[0, [2, 1, 0], :, :], f"{results_dir}/Abans_de_torch_res_{img_name}.png")

        N, C, h, w = hs.size()
        _, _, H, W = ms.size()
        _, _, H2, W2 = kks.size()

        patch_size = 48
        h_rm = h % patch_size
        h_new = h - h_rm
        H_new = H - h_rm * 2
        H2_new = H2 - h_rm // 3
        w_rm = w % patch_size
        w_new = w - w_rm
        W_new = W - w_rm * 2
        W2_new = W2 - w_rm // 3
        new_south = south + (north - south) * (h_rm / h)
        new_east = east - (east - west) * (w_rm / w)
        hs = hs[:, :, :h_new, :w_new]
        kks = kks[:, :, :H2_new, :W2_new]
        ms = ms[:, :, :H_new, :W_new]

        N, _, H, W = ms.size()
        _, C, _, _ = hs.size()

        last_folder = dataset_path.split("/")[-1]
        results_dir = f'{dataset_path}/{img_name}/Resultats'
        os.makedirs(results_dir, exist_ok=True)

        save_image(ms[0, [2, 1, 0], :, :], f"{results_dir}/Abans_de_res_{img_name}.png")

        print('Creating patches...')
        # /home/tomeugarau/BandesAPP/20250925_152412
        hs_patcher = CreatePatches(hs, patch_size, False)
        hs_patches = hs_patcher.do_patches(hs)
        # pan_patcher = CreatePatches(pan, patch_size * 2, False)
        # pan_patches = pan_patcher.do_patches(pan)
        ms_patcher = CreatePatches(ms, patch_size * 2, False)
        ms_patches = ms_patcher.do_patches(ms)

        kks_patcher = CreatePatches(kks, patch_size // 3, False)
        kks_patches = kks_patcher.do_patches(kks)
        fused = []
        kkfused = []
        # for using tqdm
        print('Patches done')

        print('Fusing image...')



        for i in tqdm(range(hs_patches.size(0))):
            hs_p = hs_patches[[i]]
            # pan_p = pan_patches[[i]]
            ms_p = ms_patches[[i]]
            kks_p = kks_patches[[i]]
            with torch.no_grad():
                hs_p = hs_p.to(device)
                # pan_p = pan_p.to(device)
                ms_p = ms_p.to(device)
                kks_p = kks_p.to(device)
                # print(f'HS before fuse: {hs_p.size()}')
                fused_p = model(hs=hs_p, pan=ms_p, ms=ms_p)
                # print(f'HS after fuse: {fused_p['pred'].size()}')
                fused_p = fused_p['pred']
                # print(f'60M before bicubic: {kks_p.size()}')
                kkfuse_p = bicubic_interpolate(kks_p, scale_factor=6)
                # print(f'60M after bicubic: {kkfuse_p.size()}')
                fused.append(fused_p.cpu())
                kkfused.append(kkfuse_p.cpu())

        fused = torch.cat(fused, dim=0)
        kkfused = torch.cat(kkfused, dim=0)

        ms_patcher.C = hs_patcher.C
        fused = ms_patcher.undo_patches(fused)
        ms_patcher.C = kks_patcher.C
        kkfused = ms_patcher.undo_patches(kkfused)
        print(f'MS shape: {ms.size()}')
        print(f'KKS shape: {kkfused.size()}')
        print(f'Fused shape: {fused.size()}')
        # data = torch.cat((fused, ms), dim=1).squeeze()
        data = torch.cat((kkfused[0,0:1], ms[0,0:1], ms[0,1:2], ms[0,2:3], fused[0,0:1], fused[0,1:2],
                          fused[0,2:3], ms[0,3:4], fused[0,3:4], kkfused[0,1:2], fused[0,4:5], fused[0,5:6]))
        data = ((2**8-1) * data.numpy()).astype(np.uint8)
        print(f'Fused shape: {data.shape}')
        count, height, width = data.shape

        print("Image fused. Now saving .tif file and image.")

        last_folder = dataset_path.split("/")[-1]
        # results_dir = f"C:/Users/Usuario/BandesAPP/Results/{last_folder}"
        results_dir = f'{dataset_path}/{img_name}/Resultats'
        os.makedirs(results_dir, exist_ok=True)
        tiff_path = f"{results_dir}/{img_name}_fused.tif"

        profile = {
            "fp": tiff_path,
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

        save_image(ms[0, [2, 1, 0], :, :], f"{results_dir}/{img_name}.png")

        messagebox.showinfo("Done", f"Fusion complete! Files saved in {results_dir}")


#A PARTIR D'AQUI INVENTOMETRO

def run_fuser():
    print("Loading checkpoints...")
    parser = argparse.ArgumentParser(description="Fuse script")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0')
    parser.add_argument('--nickname', type=str, default='All')

    args = parser.parse_args()
    device = args.device

    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)

    # Go up one level to reach the S2API folder
    project_root = os.path.dirname(src_dir)

    # Build the checkpoint path
    default_ckpt = os.path.join(project_root, 'checkpoints', 'GINet_best.ckpt')

    ckpt = torch.load(default_ckpt, map_location=args.device, weights_only=False)
    print('Checkpoints loaded')

    imgs_folder = picker.get()

    test(ckpt, args.device, args.nickname, imgs_folder)

class DirectoryPicker(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.path_var = tk.StringVar()

        # Entry to show selected directory
        self.entry = ttk.Entry(self, textvariable=self.path_var, width=40)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Browse button
        self.button = ttk.Button(self, text="Browse...", command=self.browse_directory)
        self.button.pack(side="right")

    def browse_directory(self):
        path = filedialog.askdirectory(title="Select a Folder", initialdir="C:/Users/Usuario/BandesAPP")
        if path:
            self.path_var.set(path)

    def get(self):
        """Return the currently selected directory"""
        return self.path_var.get()

# Create main window
root = tk.Tk()
root.title("GINet Sentinel-2 fuser")

# Make column expand with window resize
root.columnconfigure(0, weight=1)

# Label on top of the directory browser
label = ttk.Label(root, text="Select a S2 product directory:")
label.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")

# Directory picker widget
picker = DirectoryPicker(root)
picker.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

# Run button below
run_button = ttk.Button(root, text="Fuse selected product", command=run_fuser)
run_button.grid(row=2, column=0, pady=10)

root.mainloop()