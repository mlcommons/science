#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# stemdl_classification.py

# Stemdl Classification benchmark

# SciML-Bench
# Copyright © 2022 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

# imports from stemdl
import time, sys, os, math, glob, argparse, yaml, decimal
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# MLCommons logging
from mlperf_logging import mllog
import logging


# Custom dataset class
class NPZDataset(Dataset):
    def __init__(self, npz_root):
        self.files = glob.glob(npz_root + "/*.npz")

    def __getitem__(self, index):
        sample = np.load(self.files[index])
        x = torch.from_numpy(sample["data"])
        y = sample["label"][0]
        return (x, y)

    def __len__(self):
        return len(self.files)


# StemdlModel
class StemdlModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_size = 128
        self.num_classes = 231
        self.model_name = "resnet50"
        self.model = models.resnet50(pretrained=False)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)
        self.params_to_update = self.model.parameters()
        self.feature_extract = False

    # forward step
    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self.model(x)
        y = F.one_hot(y, num_classes=231).float()
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = self.model(x)
        y = F.one_hot(y, num_classes=231).float()
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x_hat = self.model(x)
        y = F.one_hot(y, num_classes=231).float()
        loss = F.mse_loss(x_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat


#
# Running the code: 
# python stemdl_classification.py --config stemdlConfig.yaml
#
def main():
    # Read command line arguments
    parser = argparse.ArgumentParser(description='Stemdl command line arguments', \
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default=os.path.expanduser('./stemdlConfig.yaml'), help='path to config file')
    args = parser.parse_args()

    configFile = os.path.expanduser(args.config)

    # Read YAML file
    with open(configFile, 'r') as stream:
        config = yaml.safe_load(stream)

    # MLCommons logging
    mlperf_logfile = os.path.expanduser(config['mlperf_logfile'])
    mllog.config(filename=mlperf_logfile)
    mllogger = mllog.get_mllogger()
    logger = logging.getLogger(__name__)

    # Initiase trainer object
    trainer = pl.Trainer(gpus=int(config['gpu']), num_nodes=int(config['nodes']), precision=16, strategy="ddp",
                         max_epochs=int(config['epochs']))

    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value=config['benchmark'])
        mllogger.event(key=mllog.constants.SUBMISSION_ORG, value=config['organisation'])
        mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value=config['division'])
        mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value=config['status'])
        mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value=config['platform'])
        mllogger.start(key=mllog.constants.INIT_START)

        # Values extracted from stemdlConfig.yaml
        mllogger.event(key='number_of_ranks', value=config['gpu'])
        mllogger.event(key='number_of_nodes', value=config['nodes'])
        mllogger.event(key='accelerators_per_node', value=config['accelerators_per_node'])
        mllogger.end(key=mllog.constants.INIT_STOP)
        mllogger.event(key=mllog.constants.EVAL_START, value="Start:Loading datasets")

    # Datasets
    train_dataset = NPZDataset(os.path.expanduser(config['train_dir']))
    val_dataset = NPZDataset(os.path.expanduser(config['val_dir']))
    test_dataset = NPZDataset(os.path.expanduser(config['test_dir']))
    predict_dataset = NPZDataset(os.path.expanduser(config['inference_dir']))

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config['batchsize']), num_workers=4)

    val_loader = DataLoader(dataset=val_dataset, batch_size=int(config['batchsize']), num_workers=4)

    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config['batchsize']), num_workers=4)

    predict_loader = DataLoader(dataset=predict_dataset, batch_size=int(config['batchsize']), num_workers=4)

    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_STOP, value="Stop: Loading datasets")
        mllogger.event(key=mllog.constants.EVAL_START, value="Start: Loading model")

    # Model
    model = StemdlModel()
    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_STOP, value="Stop: Loading model")

    # Training
    samples = train_dataset.__len__()
    samples_per_gpu = int(samples) / int(config['gpu'])

    start = time.time()
    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_START, value="Start: Training")

    trainer.fit(model, train_loader, val_loader)

    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_STOP, value="Stop: Training")

    diff = time.time() - start
    elapsedTime = decimal.Decimal(diff)
    training_per_epoch = elapsedTime / int(config['epochs'])
    training_per_epoch_str = f"{training_per_epoch:.2f}"

    log_file = os.path.expanduser(config['log_file'])
    if (trainer.global_rank == 0):
        with open(log_file, "a") as logfile:
            logfile.write(
                f"Stemdl training, samples_per_gpu={samples_per_gpu}, resnet={config['resnet']}, epochs={config['epochs']}, bs={config['batchsize']}, nodes={config['nodes']}, gpu={config['gpu']}, training_per_epoch={training_per_epoch_str}\n")

    # Testing
    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_START, value="Start: Testing")
    trainer.test(model, test_loader)
    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_STOP, value="Stop: Testing")

    # Inference
    number_inferences = predict_dataset.__len__()
    number_inferences_per_gpu = int(number_inferences) / (int(config['gpu']) * int(config['nodes']))
    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_START, value="Start: Inferences")

    start = time.time()
    predictions = trainer.predict(model, dataloaders=predict_loader)
    diff = time.time() - start

    if (trainer.global_rank == 0):
        mllogger.event(key=mllog.constants.EVAL_STOP, value="Stop: Inferences")
    elapsedTime = decimal.Decimal(diff)
    time_per_inference = elapsedTime / number_inferences
    time_per_inference_str = f"{time_per_inference:.6f}"

    if (trainer.global_rank == 0):
        with open(log_file, "a") as logfile:
            logfile.write(
                f"Stemdl inference, inferences_per_gpu={number_inferences_per_gpu}, bs={config['batchsize']}, nodes={config['nodes']}, gpu={config['gpu']}, time_per_inference={time_per_inference_str}\n")

        mllogger.end(key=mllog.constants.RUN_STOP, value="STEMLD benchmark run finished", metadata={'status': 'success'})


if __name__ == "__main__":
    main()
