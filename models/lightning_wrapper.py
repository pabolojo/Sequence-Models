import pytorch_lightning as pl
import torch
import numpy as np
from neural_decoder.augmentations import GaussianSmoothing

from edit_distance import SequenceMatcher

class LightningWrapper(pl.LightningModule):
    def __init__(self, model, loss_ctc, optimizer, args, scheduler=None, willetts_preprocessing_pipeline = True):
        super().__init__()
        self.save_hyperparameters()
        self.loss_ctc = loss_ctc
        self.optimizer = optimizer
        self.args = args
        self.scheduler = scheduler
        self.willetts_preprocessing_pipeline = willetts_preprocessing_pipeline

        self.validation_step_outputs = {'val_loss': [], 'val_cer': []}

        if self.willetts_preprocessing_pipeline:
            self.gaussianSmoother = GaussianSmoothing(
                args['nInputFeatures'], 20, args["gaussianSmoothWidth"], dim=1
            )
            self.dayWeights = torch.nn.Parameter(torch.randn(args['nDays'], args['nInputFeatures'], args['nInputFeatures']))
            self.dayBias = torch.nn.Parameter(torch.zeros(args['nDays'], 1, args['nInputFeatures']))

            for x in range(args['nDays']):
                self.dayWeights.data[x, :, :] = torch.eye(args['nInputFeatures'])

            # Input layers
            for x in range(args['nDays']):
                setattr(self, "inpLayer" + str(x), torch.nn.Linear(args['nInputFeatures'], args['nInputFeatures']))

            for x in range(args['nDays']):
                thisLayer = getattr(self, "inpLayer" + str(x))
                thisLayer.weight = torch.nn.Parameter(
                    thisLayer.weight + torch.eye(args['nInputFeatures'])
                )

            self.inputLayerNonlinearity = torch.nn.Softsign()

        self.model = model
        
        self.fc_decoder_out = torch.nn.Linear(args['nHiddenFeatures'], args['nClasses'] + 1)

    def batch_to_device(self, batch):
        X, y, X_len, y_len, dayIdx = batch
        X, y, X_len, y_len, dayIdx = (
            X.to(self.args['device']),
            y.to(self.args['device']),
            X_len.to(self.args['device']),
            y_len.to(self.args['device']),
            dayIdx.to(self.args['device']),
        )
        return X, y, X_len, y_len, dayIdx

    def forward(self, neuralInput, dayIdx):

        if self.willetts_preprocessing_pipeline:
            # apply gaussian smoother
            neuralInput = torch.permute(neuralInput, (0, 2, 1))
            neuralInput = self.gaussianSmoother(neuralInput)
            neuralInput = torch.permute(neuralInput, (0, 2, 1))

            # apply day layer
            dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
            transformedNeural = torch.einsum(
                "btd,bdk->btk", neuralInput, dayWeights
            ) + torch.index_select(self.dayBias, 0, dayIdx)
            neuralInput = self.inputLayerNonlinearity(transformedNeural)

        hidden = self.model(neuralInput) #transformedNeural)

        return self.fc_decoder_out(hidden)

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        X, y, X_len, y_len, dayIdx = self.batch_to_device(batch)

        # Noise augmentation is faster on GPU
        if self.args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=self.args['device']) * self.args["whiteNoiseSD"]

        if self.args["constantOffsetSD"] > 0:
            X += (
            torch.randn([X.shape[0], 1, X.shape[2]], device=self.args['device'])
            * self.args["constantOffsetSD"]
            )

        logits = self(X, dayIdx)

        loss = self.loss_ctc(
            torch.permute(logits.log_softmax(2), [1, 0, 2]),
            y,
            X_len,
            y_len,
        )
        loss = torch.sum(loss)
        
        self.log('train_loss', loss.detach())

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
    def validation_step(self, batch, batch_idx):
        total_edit_distance = 0
        total_seq_length = 0

        X, y, X_len, y_len, dayIdx = self.batch_to_device(batch)

        logits = self(X, dayIdx)

        loss = self.loss_ctc(
            torch.permute(logits.log_softmax(2), [1, 0, 2]),
            y,
            X_len,
            y_len,
        )
        loss = torch.sum(loss)

        adjustedLens = X_len
        for iterIdx in range(logits.shape[0]):
            decodedSeq = torch.argmax(
                torch.tensor(logits[iterIdx, 0 : adjustedLens[iterIdx], :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(
                y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
            )

            matcher = SequenceMatcher(
                a=trueSeq.tolist(), b=decodedSeq.tolist()
            )
            total_edit_distance += matcher.distance()
            total_seq_length += len(trueSeq)

        cer = total_edit_distance / total_seq_length

        self.log('val_loss', loss.detach())
        self.log('val_cer', cer)

        self.validation_step_outputs['val_loss'].append(loss)
        self.validation_step_outputs['val_cer'].append(cer)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs['val_loss']).mean()
        avg_acc = sum(self.validation_step_outputs['val_cer']) / len(self.validation_step_outputs['val_cer'])

        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_cer', avg_acc)

        self.validation_step_outputs['val_loss'].clear()
        self.validation_step_outputs['val_cer'].clear()

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'train_loss'  # Optional, if you want to monitor a specific metric
            }
