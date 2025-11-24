import os
from typing import Any
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.utils import save_image
from torch.nn.functional import fold, unfold
import numpy as np
import h5py


class ImageLogger(Callback):
    def __init__(self,
                 test_subsets: List = ['validation', 'test'],
                 path: str | None = None) -> None:
        super(ImageLogger, self).__init__()

        self.test_subsets = test_subsets
        self.path = path


    def on_test_epoch_start(self, trainer: "pl.Trainer",
                                  pl_module: "pl.LightningModule") -> None:

        self.path = os.path.join(pl_module.cfg.log_dir, 'images') if self.path is None else os.path.join(self.path, 'images')
        for subset in self.test_subsets:
            os.makedirs(f'{self.path}/{subset}', exist_ok=True)
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        epoch = trainer.current_epoch
        if batch_idx<2:
            dataset = trainer.val_dataloaders[dataloader_idx].dataset
            pred = outputs["pred"]
            gt  = batch["gt"]
            pan = outputs["pan"]
            data = dataset.get_rgb(pred[[0], :, :, :].detach().cpu())
            trainer.logger.experiment.add_image(f"pred_{self.test_subsets[dataloader_idx]}_{batch_idx}", data[0], global_step=epoch)
            data = dataset.get_rgb(gt[[0], :, :, :].detach().cpu())
            trainer.logger.experiment.add_image(f"gt_{self.test_subsets[dataloader_idx]}_{batch_idx}", data[0],
                                                global_step=epoch)
            data = dataset.get_rgb(pan[[0, 1, 2], :, :].detach().cpu())
            trainer.logger.experiment.add_image(f"pan_{self.test_subsets[dataloader_idx]}_{batch_idx}", data[0],
                                                global_step=epoch)

    def on_test_batch_end(self, trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule",
                                outputs: STEP_OUTPUT,
                                batch: Any,
                                batch_id: int,
                                dataloader_idx=0) -> None:

        low, gt, pan = batch['hs'], batch['gt'], batch['pan']
        batch_len = low.size(0)
        dataset = trainer.test_dataloaders[dataloader_idx].dataset
        subset = self.test_subsets[dataloader_idx]
        model = pl_module.cfg.model.name
        batch_size = pl_module.cfg.model.train.batch_size
        for key, value in outputs.items():
            for i in range(batch_len):
                save_name = f"{i+batch_size*batch_id}"
                match key:
                    case 'pred' | 'ms_lf' | 'hs_lf':
                        data = dataset.get_rgb(value[[i],:,:,:])
                        save_image(data, f"{self.path}/{subset}/{save_name}_{model}_{key}.png")
                    case 'ms_list' | 'hs_list':
                        data = dataset.get_rgb(value[-1][[i],:,:,:])
                        save_image(data[-1], f"{self.path}/{subset}/{save_name}_{model}_{key.split('_')[0]}.png")
        for i in range(batch_len):
            save_name = f"{i + batch_size * batch_id}"
            data = dataset.get_rgb(gt[[i],:,:,:])
            save_image(data, f"{self.path}/{subset}/{save_name}_gt.png")

class ImageSaver(Callback):
    def __init__(self,name) -> None:
        super(ImageSaver, self).__init__()
        self.pred_images = []
        self.ms_images = []
        self.gt_images = []
        self.name = name

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pred = outputs['pred']
        self.pred_images.append(pred.cpu())
        self.ms_images.append(batch['ms'].cpu())
        self.gt_images.append(batch['gt'].cpu())

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        fused = torch.cat(self.pred_images, dim=0)
        ms = torch.cat(self.ms_images, dim=0)
        gt = torch.cat(self.gt_images, dim=0)
        print(ms.size())
        print(gt.size())
        fused = self.undo_patches(fused, 1200, 1200)
        gt = self.undo_patches(gt, 1200, 1200)
        ms = self.undo_patches(ms, 1200, 1200)

        # print(f'Fused size: {fused.size()}')
        # print(f'Data size: {data.size()}')

        ms_band_names = ["B2", "B3", "B4", "B8"]
        fused_band_names = ["B5", "B6", "B7", "B8A", "B11", "B12"]

        save_dir = f'/home/dani/projects/MaLiSatDetection/results/Sentinel2imgs_ivan/{pl_module.cfg.model.name}'
        os.makedirs(save_dir, exist_ok=True)

        with h5py.File(f"{save_dir}/{pl_module.cfg.model.name}_{self.name}_{pl_module.cfg.nickname}.he5", "w") as f:
            f.create_dataset("ms", data=ms)
            f["ms"].attrs["bands"] = ",".join(ms_band_names)
            f.create_dataset("fused", data=fused)
            f["fused"].attrs["bands"] = ",".join(fused_band_names)

        with h5py.File(f"{save_dir}/GT_{self.name}.he5", "w") as f:
            f.create_dataset("ms", data=ms)
            f["ms"].attrs["bands"] = ",".join(ms_band_names)
            f.create_dataset("fused", data=gt)
            f["fused"].attrs["bands"] = ",".join(fused_band_names)

    # def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #
    #     fused = torch.cat(self.pred_images, dim=0)
    #     ms = torch.cat(self.ms_images, dim=0)
    #     gt = torch.cat(self.gt_images, dim=0)
    #
    #     fused = self.undo_patches(fused, 1200, 1200)
    #     ms = self.undo_patches(ms, 1200, 1200)
    #
    #     # print(f'Fused size: {fused.size()}')
    #     # print(f'Data size: {data.size()}')
    #
    #     ms_band_names = ["B2", "B3", "B4", "B8"]
    #     fused_band_names = ["B5", "B6", "B7", "B8A", "B11", "B12"]
    #
    #     save_dir = f'/home/dani/projects/MaLiSatDetection/results/Sentinel2imgspan/{pl_module.cfg.model.name}'
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     with h5py.File(f"{save_dir}/GT_{self.name}.he5", "w") as f:
    #         f.create_dataset("ms", data=ms)
    #         f["ms"].attrs["bands"] = ",".join(ms_band_names)
    #         f.create_dataset("fused", data=fused)
    #         f["fused"].attrs["bands"] = ",".join(fused_band_names)

    @staticmethod
    def undo_patches(data, H, W):
        #patches = data.reshape(1, data.size(0), data.size(1), data.size(2), data.size(2))
        #patches = patches.view(1, data.size(0), data.size(1) * data.size(2) * data.size(2)).permute(0, 2, 1)
        #return fold(patches, (W, H), kernel_size=data.size(2), stride=data.size(2))
        ps = data.size(2)
        overlap = 60
        N = W // overlap - 1
        M = len(data) // N
        patchwork = torch.zeros((1, data.size(1), H, W)).to(data.device)
        for patch_idx, patch in enumerate(data):
            i_start = (patch_idx // ((W - overlap) // (ps - overlap))) * (ps - overlap)
            j_start = (patch_idx % ((W - overlap) // (ps - overlap))) * (ps - overlap)
            i_end = i_start + ps
            j_end = j_start + ps
            imin = i_start + overlap // 2
            imax = i_end - overlap // 2
            iimin = overlap // 2
            iimax = ps - overlap // 2
            jmin = j_start + overlap // 2
            jmax = j_end - overlap // 2
            jjmin = overlap // 2
            jjmax = ps - overlap // 2
            if patch_idx < N:
                imin = i_start
                iimin = 0
            if patch_idx % N == 0:
                jmin = j_start
                jjmin = 0
            if patch_idx % N == N - 1:
                jmax = j_end
                jjmax = ps
            if patch_idx >= N * (M - 1):
                imax = i_end
                iimax = ps
            a = patchwork[0, :, imin:imax, jmin:jmax]
            b = patch[:, iimin:iimax, jjmin:jjmax]
            # if patch_idx== 210:
            #     continue
            patchwork[0, :, imin:imax, jmin:jmax] = patch[:, iimin:iimax, jjmin:jjmax]
        return patchwork

