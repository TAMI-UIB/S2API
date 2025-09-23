from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class GradientCheck(Callback):
    def __init__(self, ) -> None:
        super(GradientCheck, self).__init__()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        epoch = trainer.current_epoch
        if epoch == 0 and batch_id==0:
            print("Checking learnable parameters gradients")
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"[WITHOUT GRADIENT] {name}")
        # else:
        #     for name, param in pl_module.named_parameters():
        #         if param.requires_grad and param.grad is None:
        #             print(f"[WITHOUT GRADIENT] {name}")

