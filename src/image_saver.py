import torch
import h5py
from torchvision.utils import save_image
import numpy as np
import os
import argparse

def load_composite_from_he5(filepath, composite_type):
    with h5py.File(filepath, "r") as f:
        ms = torch.from_numpy(f["ms"][:])                # [1, C1, H, W]
        up20 = torch.from_numpy(f["fused"][:])   # [1, C2, H, W]

        ms_bands = f["ms"].attrs["bands"].split(",")
        up_bands = f["fused"].attrs["bands"].split(",")

    # Create band dictionary for lookup
    band_dict = {name: ms[0, i] for i, name in enumerate(ms_bands)}
    band_dict.update({name: up20[0, i] for i, name in enumerate(up_bands)})

    # Define composite presets
    presets = {
        "rgb": ["B4", "B3", "B2"],
        "urban": ["B12", "B11", "B4"],
        "rural": ["B5", "B6", "B7"],
        "coastal": ["B3", "B8A"],
        "rural_index": ["B8A", "B11"],
        "swit": ["B12","B8A", "B4"]   }

    selected_bands = presets[composite_type]

    if composite_type == "coastal":
        bands = torch.stack([band_dict[b] for b in selected_bands], dim=0)
        ndwi = (bands[0:1]-bands[1:2])/(bands[0:1]+bands[1:2])
        return (ndwi+1)/2

    elif composite_type == "rural_index":
        bands = torch.stack([band_dict[b] for b in selected_bands], dim=0)
        ndwi = (bands[0:1]-bands[1:2])/(bands[0:1]+bands[1:2])
        return  (ndwi+1)/2

    else:
        rgb = torch.stack([band_dict[b] for b in selected_bands], dim=0)  # [3, H, W]
        return rgb.clamp(0,1)

    # Normalize per-band to [0, 1] for visualization
    # rgb = (rgb - rgb.min(dim=(1,2), keepdim=True)[0]) / \
    #       (rgb.max(dim=(1,2), keepdim=True)[0] - rgb.min(dim=(1,2), keepdim=True)[0] + 1e-6)

from pathlib import Path


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RGB composite from Sentinel-2 .he5 file")
    parser.add_argument("--model", default="finetune")
    parser.add_argument("--filename", default="finetune_Barcelona_dbumlp_loss", help="Base name of the .he5 file (without extension)")
    parser.add_argument("--composite", default="urban", choices=["rgb", "urban", "rural", "coastal"],
                        help="Type of composite to generate")
    args = parser.parse_args()

    input_dir = f'/home/dani/projects/MaLiSatDetection/results/Sentinel2_ivan'
    dir = Path(input_dir)
    files = list(dir.glob("*.he5"))
    out_dir = f'/home/dani/projects/MaLiSatDetection/results/results_ivan'
    for file in files:

        model, place = file.stem.split("_")
        if place == "NovaZelanda":
            composite = "swit"
        if place == "Barcelona":
            composite = "coastal"
        rgb = load_composite_from_he5(file, composite)
        os.makedirs(f"{out_dir}/{place}", exist_ok=True)
        save_image(rgb, f"{out_dir}/{place}/{model}.png")


    # dir_list = os.listdir(f'/home/dani/projects/MaLiSatDetection/results/Sentinel2imgs/')
    # output_file = f"{args.filename}_{args.composite}.png"
    #

    # os.makedirs(save_dir, exist_ok=True)
    #
    #
    # rgb = load_composite_from_he5(input_file, args.composite)
    # save_image(rgb, f'{save_dir}/{output_file}')
    #
    # print(f"{args.composite} composite saved as {output_file}")