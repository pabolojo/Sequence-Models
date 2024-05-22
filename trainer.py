#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!cd $HOME/speechBCI/NeuralDecoder && pip install --user -e .
#!cd $HOME/speechBCI/LanguageModelDecoder/runtime/server/x86 && python setup.py install
#!pip install causal-conv1d
#!cd $HOME/mamba && pip install --user -e .
#!cd $HOME/neural_seq_decoder && pip install --user -e .
#!pip install pytorch-lightning
#!pip install tensorboard


# ### Imports and Script vars

# In[2]:


#%load_ext autoreload
#%autoreload 2


# In[3]:


import torch
import pickle
import os
import time
import numpy as np
import pytorch_lightning as pl
import sys

from models.datasetLoaders import getDatasetLoaders
from models.mamba_phoneme import MambaPhoneme
from models.lightning_wrapper import LightningWrapper
from mamba_ssm.models.config_mamba import MambaConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

def train(confPath):
    # Load the configuration pickle file
    print("Loading configuration from: ", confPath, flush=True)
    args = pickle.load(open(confPath, 'rb'))

    ssm_cfg = {
            'd_state'   : args["d_state"],
            'd_conv'    : args["d_conv"],
            'expand'    : args["expand"],
            'dt_rank'   : args["dt_rank"],
            'dt_min'    : args["dt_min"],
            'dt_max'    : args["dt_max"],
            'dt_init'   : args["dt_init"],
            'dt_scale'  : args["dt_scale"],
            'dt_init_floor' : args["dt_init_floor"],
            'conv_bias' : args["conv_bias"],
            'bias'      : args["bias"],
            'use_fast_path' : args["use_fast_path"],  # Fused kernel options
            }

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])


    # ### Load Datasets

    # In[5]:


    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args['dataset'], args['batchSize']
    )

    args['nDays'] = len(loadedData["train"])


    # ### Initialize model

    # In[6]:


    coreModel = MambaPhoneme(
        config=MambaConfig(
            d_model=args['nInputFeatures'],
            n_layer=args['nLayers'],
            vocab_size=args['nClasses'],
            ssm_cfg=ssm_cfg,
            rms_norm=False,
            residual_in_fp32=False,
            fused_add_norm=False,
        ),
        device=args['device'],
        dtype=torch.float32,
    )


    # In[7]:


    print(coreModel.modelName)
    print('Number of parameters: ', sum(p.numel() for p in coreModel.parameters() if p.requires_grad))
    print('\n--------------------\n')
    print(coreModel)
    print('\n--------------------\n')


    # ### Train

    # In[8]:


    # Set seeds and setup output directory
    timestamp = int(time.time())
    logsPath = args['baseDir'] + "experiments/logs"
    checkpointPath = args['experimentPath'] + "/checkpoints/" + args['modelName'] + "_" + str(timestamp)

    # Define the logger
    #logger = TensorBoardLogger(logsPath, name=coreModel.modelName)
    wandb_logger = WandbLogger(project='PNPL', name=args['modelName'], log_model='all', save_dir=logsPath)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointPath,
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}-{avg_val_cer:.2f}',
        save_last=False,
        save_top_k=2,
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

    callbacks = [checkpoint_callback, early_stop_callback]

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = torch.optim.Adam(
        coreModel.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nEpochs"],
    )

    # Training
    model = LightningWrapper(coreModel, loss_ctc, optimizer, args, scheduler, willetts_preprocessing_pipeline = args['pppipeline'])


    # In[9]:


    trainer = pl.Trainer(
        max_epochs=args["nEpochs"],
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=False
    )

    trainer.fit(model, trainLoader, val_dataloaders=testLoader)


# In[ ]:

if __name__ == "__main__":
    if len(sys.argv) > 1:
        confPath = sys.argv[1]
        train(confPath)

