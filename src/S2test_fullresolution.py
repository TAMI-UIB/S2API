import os
import argparse


from datetime import datetime
import hydra
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from base import Experiment
from src.utils.callbacks.loss_logger import LossLogger
from src.utils.callbacks.metric_logger import MetricLogger
from src.utils.callbacks.image_logger import ImageLogger, ImageSaver


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def test(cfg: DictConfig):
    data_loader = instantiate(cfg.dataset.datamodule)
    nickname = cfg.nickname
    city = cfg.dataset.datamodule.test_path
    ckpt = torch.load(f"{cfg.log_path}/checkpoints/best.ckpt", map_location=f"cuda:{cfg.devices[0]}", weights_only=False)
    cfg = ckpt["cfg"]
    experiment = Experiment(cfg)

    callback_list = [ImageSaver(name=city)]

    trainer = Trainer(max_epochs=cfg.model.train.max_epochs,
                      devices=cfg.devices,
                      callbacks=callback_list)

    experiment.load_state_dict(ckpt['state_dict'], strict=True)
    trainer.test(experiment, dataloaders=data_loader)

if __name__ == '__main__':
    test()
