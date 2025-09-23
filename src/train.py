import os

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
from src.utils.callbacks.image_logger import ImageLogger
from src.utils.callbacks.gradient_check import GradientCheck

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)
    cfg.update({"day": datetime.now().strftime("%Y-%m-%d")})
    cfg.update({"log_dir": f'{os.environ["PROJECT_ROOT"]}/logs/{cfg.dataset.name}/{cfg.day}/{cfg.model.name}/'})

    data_loader = instantiate(cfg.dataset.datamodule)
    experiment = Experiment(cfg)
    logger = TensorBoardLogger(
        f'{os.environ["PROJECT_ROOT"]}/logs/{cfg.dataset.name}/{cfg.day}/{cfg.model.name}/',
        name=cfg.nickname)
    cfg.update({"log_dir": logger.log_dir})

    default_callbacks = [
                            LossLogger(),
                            MetricLogger(),
                            # ImageLogger(),
                            RichModelSummary(max_depth=4),
                            GradientCheck(),
                            instantiate(cfg.checkpoint),

                         ]

    callback_list = instantiate(cfg.model.callbacks) + default_callbacks if hasattr(cfg.model, 'callbacks') else default_callbacks
    trainer = Trainer(max_epochs=cfg.model.train.max_epochs, logger=logger,
                      devices=cfg.devices,
                      callbacks=callback_list,
                      )

    trainer.fit(experiment, train_dataloaders=data_loader)
    ckpt = torch.load(f'{cfg.log_dir}/checkpoints/best.ckpt', map_location=f'cuda:{cfg.devices[0]}')
    experiment.load_state_dict(ckpt['state_dict'])
    trainer.test(experiment, datamodule=data_loader)

if __name__ == '__main__':
    train()
