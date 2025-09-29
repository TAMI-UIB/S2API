# The MaLiSat Toolbox 🌊🛰️

**Pre-Alpha**

Sea2Net is a research tool designed to streamline the workflow of working with Sentinel-2 (S2) satellite imagery for marine applications.  
Given a set of coordinates and a time period, the app:

1. **Fetches Sentinel-2 products** available over the region and time span.  
2. **Upsamples the 20 m bands** to 10 m resolution using a guided-image super-resolution network (**GINet**).

   The GINet is the architecture proposed in _Super-Resolution of Sentinel-2 Images Using a Geometry-Guided Back-Projection Network with Self-Attention_, available on arXiv:

    [![arXiv](https://img.shields.io/badge/arXiv-2409.02675-B31B1B.svg)](https://arxiv.org/abs/2508.04729)
3. *(Planned)* Feeds the enhanced images into a segmentation network to **detect marine litter**.


## Features

- 📍 Input geographical coordinates  
- ⏳ Choose a time interval  
- ⬇️ Automatic download of Sentinel-2 products  
- 🔍 Super-resolution of 20 m bands with GINet  
- 🧪 Early experimental stage (pre-alpha)

---

## User Guide

Follow these steps to get started with the **MaLiSat Toolbox**

### 1. Clone the repository
```bash
git clone https://github.com/TAMI-UIB/S2API.git
cd S2API
```

### 2. Create a virtual environment (optional but recommended)
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
⚠️ Make sure your Python version is compatible (Python 3.10 or higher recommended).

### 4. Run the Launcher
```bash
python launcer.py
```
* The launcher will open a GUI asking whether to download new Sentinel-2 products or fuse existing products (in the following version, a marine litter detection option will be added).
* For downloading, you’ll enter coordinates, select a date range, set max cloud cover, and choose a save directory.
* If products are found, the app will ask how many to download.
* For fusing, you just select the folder containing previously downloaded products.

### 5. Notes
* Internet connection required for downloading Sentinel-2 imagery.
* The checkpoints folder must be present for the fuser (```checkpoints/GINet_best.ckpt```). The default path is automatically detected relative to the repo.
* Output files are saved in the chosen folder (default: ```~/BandesAPP/```).
* Running on GPU is strongly recommended.

---

## Roadmap / To-Dos

- [ ] Integrate segmentation network for marine litter detection  
- [ ] Improve data handling for large areas and long time spans  
- [ ] Build a user-friendly interface (CLI / web app)  
- [ ] Add unit tests and benchmarking against baseline methods  
