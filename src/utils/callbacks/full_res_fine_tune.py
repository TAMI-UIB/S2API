import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.optim import Adam
from pathlib import Path
import shutil

class FullResFineTune(Callback):
    def __init__(self, ) -> None:
        super(FullResFineTune, self).__init__()


    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        cfg = pl_module.cfg
        ckpt = torch.load(cfg.ckpt_path, weights_only=False)
        pl_module.load_state_dict(ckpt["state_dict"], strict=False)
        pl_module.model.full_res_fine_tune()
