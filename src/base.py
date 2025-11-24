import os
from typing import Dict, Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.utils import save_image

pl.seed_everything(42)


class Experiment(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        # Experiment configuration
        self.cfg = cfg
        # Define subsets
        self.subsets = ['train', 'validation', 'test']
        self.fit_subsets = ['train', 'validation']
        # Define model and loss
        self.model = instantiate(cfg.model.module)

        self.loss_criterion = instantiate(cfg.model.train.loss)
        # Number of model parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Metric calculator
        #self.metrics = {k: instantiate(cfg.metrics) for k in self.subsets}
        # Loss report
        #self.loss = {subset: 0 for subset in self.fit_subsets}

    def forward(self, pan, hs, ms):

        return self.model(pan=pan, hs=hs, ms=ms)

    def training_step(self, input, idx):

        output = self.forward(pan=input['pan'], hs=input['hs'], ms=input['ms'])
        loss = self.loss_criterion(output, input['gt'])
        #self.loss_report(loss.item(), 'train')

        return {"loss": loss, "output": output}

    def validation_step(self, input, idx, dataloader_idx=0):
        output = self.forward(pan=input['pan'], hs=input['hs'], ms=input['ms'])
        loss = self.loss_criterion(output, input['gt'])
        #self.loss_report(loss.item(), 'validation')
        return output

    def test_step(self, input, idx,  dataloader_idx=0):
        output = self.forward(pan=input['pan'], hs=input['hs'], ms=input['ms'])
        return output

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.train.optimizer,params=self.parameters())
        scheduler = instantiate(self.cfg.model.train.scheduler, optimizer=optimizer)
        return {'optimizer': optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['cfg'] = self.cfg
        checkpoint['current_epoch'] = self.current_epoch

    def loss_report(self, loss, subset):
        self.loss[subset] += loss