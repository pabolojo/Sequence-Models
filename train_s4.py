from argparse import ArgumentParser
import dotenv
dotenv.load_dotenv()

import yaml
import os
import time
import numpy as np
import pytorch_lightning as pl
import sys

from models.datasetLoaders import getDatasetLoaders
from models.s4lightning import S4Lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner

def train(args):
    # Load the hparams yaml file
    hparamsPath = args.configs
    print("Loading hparams from: ", hparamsPath, flush=True)
    with open(hparamsPath, 'r') as f:
        hparams = yaml.safe_load(f)
    
    args = vars(args)
    #delete none args
    args = {k: v for k, v in args.items() if v is not None}
    hparams.update(args)

    seed_everything(hparams["seed"], workers=True)
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    hparams['modelName'] = hparams['modelType'] + "_" + date + "_"+ str(hparams['nLayers']) + "_layers_" + str(hparams['nHiddenFeatures']) + "_d_state_" + str(hparams['batchSize']) + "_bs_" + str(hparams['lr']) + "_lr_" + str(hparams['nEpochs']) + "_epochs"

    # Training
    model = S4Lightning(hparams)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        hparams['dataset'], hparams['batchSize']
    )

    hparams['nDays'] = len(loadedData["train"])


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


    # Define the logger
    #logger = TensorBoardLogger(logsPath, name=coreModel.modelName)

    wandb_logger = None
    if hparams['log_wandb']:
        wandb_logger = WandbLogger(project='PNPL', name=hparams['modelName'], save_dir=logsPath)
        wandb_logger.log_hyperparams(hparams)
    trainer = pl.Trainer(
        max_epochs=hparams["nEpochs"],
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        log_every_n_steps=10,
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
    parser = ArgumentParser(description="Hyperparameters for our experiments")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--configs", type=str, default="configs/s4.yaml", help="Path to the config file")
    parser.add_argument("--log_wandb", type=bool, help="Log to wandb")
    parser.add_argument("--lrEnd", type=float, help="End learning rate for LR finder")
    parser.add_argument("--nLayers", type=int, help="Number of layers")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility")
    args = parser.parse_args()
    train(args)
