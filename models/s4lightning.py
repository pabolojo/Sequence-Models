import pytorch_lightning as pl
import torch
import numpy as np
from neural_decoder.augmentations import GaussianSmoothing 
from models.s4backbone import S4Backbone

from edit_distance import SequenceMatcher

class S4Lightning(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.validation_step_outputs = {'val_loss': [], 'val_cer': []}

        if self.hparams.pppipeline:
            self.gaussianSmoother = GaussianSmoothing(
                self.hparams.nInputFeatures, 20, self.hparams.gaussianSmoothWidth, dim=1
            )
            self.dayWeights = torch.nn.Parameter(torch.randn(self.hparams.nDays, self.hparams.nInputFeatures, self.hparams.nInputFeatures))
            self.dayBias = torch.nn.Parameter(torch.zeros(self.hparams.nDays, 1, self.hparams.nInputFeatures))

            for x in range(self.hparams.nDays):
                self.dayWeights.data[x, :, :] = torch.eye(self.hparams.nInputFeatures)

            # Input layers
            for x in range(self.hparams.nDays):
                setattr(self, "inpLayer" + str(x), torch.nn.Linear(self.hparams.nInputFeatures, self.hparams.nInputFeatures))

            for x in range(self.hparams.nDays):
                thisLayer = getattr(self, "inpLayer" + str(x))
                thisLayer.weight = torch.nn.Parameter(
                    thisLayer.weight + torch.eye(self.hparams.nInputFeatures)
                )

            self.inputLayerNonlinearity = torch.nn.Softsign()

        if self.hparams.modelType == "s4":
            self.coreModel = S4Backbone(
                d_input=self.hparams.nInputFeatures,
                d_model=self.hparams.nHiddenFeatures,
                d_output=self.hparams.nHiddenFeatures,
                n_layers=self.hparams.nLayers,
                dropout=0.2,
                prenorm=False,
                lr=self.hparams.lr,
            )
        
        self.fc_decoder_out = torch.nn.Linear(self.hparams.nHiddenFeatures, self.hparams.nClasses + 1)

        print("Number of parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def batch_to_device(self, batch):
        X, y, X_len, y_len, dayIdx = batch
        X, y, X_len, y_len, dayIdx = (
            X.to(self.hparams.device),
            y.to(self.hparams.device),
            X_len.to(self.hparams.device),
            y_len.to(self.hparams.device),
            dayIdx.to(self.hparams.device),
        )
        return X, y, X_len, y_len, dayIdx

    def forward(self, neuralInput, dayIdx):

        if self.hparams.pppipeline:
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

        hidden = self.coreModel(neuralInput)

        return self.fc_decoder_out(hidden)

    def training_step(self, batch, batch_idx):

        X, y, X_len, y_len, dayIdx = self.batch_to_device(batch)

        # Noise augmentation is faster on GPU
        if self.hparams.whiteNoiseSD > 0:
            X += torch.randn(X.shape, device=self.hparams.device) * self.hparams.whiteNoiseSD

        if self.hparams.constantOffsetSD > 0:
            X += (
            torch.randn([X.shape[0], 1, X.shape[2]], device=self.hparams.device)
            * self.hparams.constantOffsetSD
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

        return loss
    
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
                logits[iterIdx, 0 : adjustedLens[iterIdx], :],
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

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs['val_loss']).mean()
        avg_acc = sum(self.validation_step_outputs['val_cer']) / len(self.validation_step_outputs['val_cer'])

        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_cer', avg_acc)

        self.validation_step_outputs['val_loss'].clear()
        self.validation_step_outputs['val_cer'].clear()

    def configure_optimizers(self): 
        
        all_parameters = list(self.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weightDecay,)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )


        if self.hparams.useScheduler:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.hparams.lrEnd / self.hparams.lr,
                total_iters=100
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
