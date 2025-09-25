# The MaLiSat Toolbox 🌊🛰️

**Pre-Alpha**

Sea2Net is a research tool designed to streamline the workflow of working with Sentinel-2 (S2) satellite imagery for marine applications.  
Given a set of coordinates and a time period, the app:

1. **Fetches Sentinel-2 products** available over the region and time span.  
2. **Upsamples the 20 m bands** to 10 m resolution using a guided-image super-resolution network (**GINet**).

   The GINet is the architecture proposed in _Super-Resolution of Sentinel-2 Images Using a Geometry-Guided Back-Projection Network with Self-Attention_, available on arXiv:

    [![arXiv](https://img.shields.io/badge/arXiv-2409.02675-B31B1B.svg)](https://arxiv.org/abs/2508.04729)
3. *(Planned)* Feeds the enhanced images into a segmentation network to **detect marine litter**.

---

## Features

- 📍 Input geographical coordinates  
- ⏳ Choose a time interval  
- ⬇️ Automatic download of Sentinel-2 products  
- 🔍 Super-resolution of 20 m bands with GINet  
- 🧪 Early experimental stage (pre-alpha)

---

## Roadmap / To-Dos

- [ ] Integrate segmentation network for marine litter detection  
- [ ] Improve data handling for large areas and long time spans  
- [ ] Build a user-friendly interface (CLI / web app)  
- [ ] Add unit tests and benchmarking against baseline methods  
