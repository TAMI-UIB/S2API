import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.optim import Adam
from pathlib import Path
import shutil

class PostFineTune(Callback):
    def __init__(self, max_epochs, loss_function, lr) -> None:
        super(PostFineTune, self).__init__()
        self.epoch = 2000
        self.loss_function = loss_function
        self.lr =lr
        print(self.epoch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_epoch = trainer.current_epoch
        if self.epoch == current_epoch:
            cfg = pl_module.cfg
            log_path = f"{cfg.log_dir}/checkpoints/best.ckpt"
            self._copy_weights(cfg.log_dir)
            ckpt = torch.load(log_path, weights_only=False)
            pl_module.load_state_dict(ckpt["state_dict"])
            pl_module.model.fine_tune(current_epoch)
            pl_module.loss_criterion = self.loss_function
            pl_module.optimizers =  [Adam(pl_module.model.parameters(), lr=self.lr)]
            trainable_params = sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in pl_module.model.parameters())
            print(f"Number of parameters trainable parameters: {trainable_params}/{total_params}")

    def _copy_weights(self, dir):
        src_path = f"{dir}/checkpoints/best.ckpt"
        src = Path(src_path)
        target_path = f"{dir}/checkpoints/best_pretrained.ckpt"
        target = Path(target_path)
        shutil.copy(src,target)