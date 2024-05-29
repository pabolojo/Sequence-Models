import dotenv
dotenv.load_dotenv()

import torch
import yaml
import os
import time
import numpy as np
import pytorch_lightning as pl
import sys

from models.datasetLoaders import getDatasetLoaders
from models.lightning_wrapper import LightningWrapper
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner

def train(hparamsPath):
    # Load the hparams yaml file
    print("Loading hparams from: ", hparamsPath, flush=True)
    with open(hparamsPath, 'r') as f:
        hparams = yaml.safe_load(f)

    seed_everything(hparams["seed"], workers=True)

    # ### Load Datasets

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        hparams['dataset'], hparams['batchSize']
    )

    hparams['nDays'] = len(loadedData["train"])

    # ### Train

    # Set seeds and setup output directory
    timestamp = int(time.time())
    logsPath = os.path.join(hparams['experimentPath'], "logs")
    os.makedirs(logsPath, exist_ok=True)
    checkpointPath = hparams['experimentPath'] + "/checkpoints/" + hparams['modelName'] + "_" + str(timestamp)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointPath,
        monitor='avg_val_cer',
        filename='{epoch:02d}-{val_loss:.2f}-{avg_val_cer:.2f}',
        save_last=False,
        save_top_k=1,
        verbose=True,
        mode='min'
    )

    # Define early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min'
    )

    callbacks = [checkpoint_callback] #, early_stop_callback]

    # Training
    model = LightningWrapper(hparams)

    # Define the logger
    #logger = TensorBoardLogger(logsPath, name=coreModel.modelName)
    wandb_logger = None
    if hparams['log_wandb']:
        wandb_logger = WandbLogger(project='PNPL', name=hparams['modelName'], log_model='all', save_dir=logsPath)
        wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        max_epochs=hparams["nEpochs"],
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=False,
    )

    if hparams['lrFinder']:
        print("Running LR finder", flush=True)

        tuner = Tuner(trainer)

        lr_finder = tuner.lr_find(
            model,
            trainLoader,
            val_dataloaders=testLoader,
            min_lr=hparams["lrEnd"],
            max_lr=hparams["lr"],
            num_training=10,
            mode="exponential",
            early_stop_threshold=2.0)

        print("Suggested learning rate: ", lr_finder.suggestion(), flush=True)

        model.hparams.lr = lr_finder.suggestion()
    
    if hparams['resume']:
        print("Resuming from checkpoint: ", hparams['resume'], flush=True)

    trainer.fit(model, trainLoader, val_dataloaders=testLoader, ckpt_path=hparams['resume'] if 'resume' in hparams else None)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train(sys.argv[1])
