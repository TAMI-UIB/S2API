import csv
import os

from typing import Any

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from src.utils.metrics import MetricCalculator

class MetricLogger(Callback):
    def __init__(self, city_name="Soller", nickname="cnmf", metric_monitor='PSNR', subsets=['train', 'validation', 'test'], validation_subsets=['validation', 'test'], metric_list=['PSNR', 'SSIM', 'SAM', 'ERGAS'], path=None) -> None:
        super(MetricLogger, self).__init__()

        self.metric_list = metric_list
        self.metric_monitor = metric_monitor


        self.validation_subsets = ['validation', 'test']
        self.subsets = ['train', 'validation', 'test']
        self.test_subsets = ['test']
        self.city_name = city_name
        self.nickname = nickname
        self.metrics = {k: MetricCalculator(metric_list) for k in subsets}

        self.path = path


    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.metrics['train'].clean()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for subset in self.validation_subsets:
            self.metrics[subset].clean()

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for subset in self.test_subsets:
            self.metrics[subset].clean()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        writer = trainer.logger.experiment
        epoch = trainer.current_epoch
        metric_means = {subset: self.metrics[subset].get_means() for subset in self.subsets}
        for subset in self.subsets:
            for metric in self.metric_list:
                writer.add_scalar(f"{metric}/{subset}", metric_means[subset][metric], epoch)
        for metric in self.metric_list:
            writer.add_scalars(f"{metric}/comparison", {k: v[metric] for k, v in metric_means.items()}, epoch)
        pl_module.log(f'{self.metric_monitor}', metric_means['validation'][self.metric_monitor],  prog_bar=True)


    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        gt = batch['gt']
        self.metrics['train'].update(preds=outputs['output']['pred'], targets=gt)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:

        gt = batch['gt']
        subset = self.validation_subsets[dataloader_idx]
        self.metrics[subset].update(preds=outputs['pred'], targets=gt)


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        gt = batch['gt']
        subset = self.test_subsets[dataloader_idx]
        self.metrics[subset].update(preds=outputs['pred'], targets=gt)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        writer = trainer.logger.experiment
        epoch = trainer.current_epoch
        metric_means = {subset: self.metrics[subset].get_means() for subset in self.test_subsets}
        print(metric_means)
        for subset, mean_metrics in metric_means.items():
            new_dict = {"city": self.city_name} | {k: v for k, v in metric_means["test"].items()}
            nombre_archivo = f"{os.environ['PROJECT_ROOT']}/csv_log/{self.nickname}.csv"
            archivo_existe = os.path.exists(nombre_archivo)
            with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as archivo_csv:
                escritor = csv.DictWriter(archivo_csv, fieldnames=new_dict.keys())
                # Si el archivo no existe, escribe la cabecera
                if not archivo_existe:
                    escritor.writeheader()
                # Escribe una nueva fila
                escritor.writerow(new_dict)
            for name, value in mean_metrics.items():
                pl_module.log(f"{subset}_{name}", value)

        for subset in self.test_subsets:
            for metric in self.metric_list:
                writer.add_scalar(f"{metric}/{subset}", metric_means[subset][metric], epoch)
        for metric in self.metric_list:
            writer.add_scalars(f"{metric}/comparison", {k: v[metric] for k, v in metric_means.items()}, epoch)
        pl_module.log(f'{self.metric_monitor}', metric_means['test'][self.metric_monitor],  prog_bar=True)

    # def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     # Definir el path del directorio de reportes
    #     path = os.path.join(pl_module.cfg.log_dir, 'metric_report') if self.path is None else self.path
    #     os.makedirs(path, exist_ok=True)
    #
    #     # Datos comunes a todos los subsets
    #     base_data = {
    #         "day": str(pl_module.cfg.day),
    #         "model": pl_module.cfg.model.name,
    #         "nickname": pl_module.cfg.nickname,
    #         "log_dir": path,
    #     }
    #     for subset in self.validation_subsets:
    #         # Preparar datos de medias
    #         means = self.metrics[subset].get_means()
    #         mean_data = pd.DataFrame({**base_data, **means},
    #                                  index=[0])  # Crear con index=[0] para alinearse con el formato
    #
    #         # Preparar datos de todas las métricas
    #         all_metrics = self.metrics[subset].get_dict()
    #         n_img = len(all_metrics[self.metric_list[0]])
    #
    #         # Datos repetidos y todas las métricas
    #         all_data = pd.DataFrame({
    #             "day": [pl_module.cfg.day] * n_img,
    #             "model": [pl_module.cfg.model.name] * n_img,
    #             "nickname": [pl_module.cfg.nickname] * n_img,
    #             "log_dir": [path] * n_img,
    #             "img_id": list(range(n_img)),
    #             **{key: all_metrics[key] for key in self.metric_list}
    #         })
    #
    #         # Guardar o concatenar los archivos CSV
    #         mean_file_path = f'{path}/{subset}_mean.csv'
    #         all_file_path = f'{path}/{subset}_all.csv'
    #
    #         if self.path is None:
    #             mean_data.to_csv(mean_file_path, index=False)
    #             all_data.to_csv(all_file_path, index=False)
    #         else:
    #             self.save_or_append_csv(mean_file_path, mean_data)
    #             self.save_or_append_csv(all_file_path, all_data)

    @staticmethod
    def save_or_append_csv(file_path, new_data):
        try:
            old_data = pd.read_csv(file_path)
            pd.concat([old_data, new_data]).to_csv(file_path, index=False)
        except FileNotFoundError:
            new_data.to_csv(file_path, index=False)