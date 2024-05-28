import pytorch_lightning as pl
import torch
import numpy as np
from neural_decoder.augmentations import GaussianSmoothing
from models.mamba_phoneme import MambaPhoneme
from mamba_ssm.models.config_mamba import MambaConfig

from edit_distance import SequenceMatcher

class LightningWrapper(pl.LightningModule):
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

        if self.hparams.modelType == "mamba":
            self.coreModel = MambaPhoneme(
                config=MambaConfig(
                    d_model=self.hparams.nInputFeatures,
                    n_layer=self.hparams.nLayers,
                    vocab_size=self.hparams.nClasses,
                    ssm_cfg={
                        'd_state'   : self.hparams.d_state,
                        'd_conv'    : self.hparams.d_conv,
                        'expand'    : self.hparams.expand,
                        'dt_rank'   : self.hparams.dt_rank,
                        'dt_min'    : self.hparams.dt_min,
                        'dt_max'    : self.hparams.dt_max,
                        'dt_init'   : self.hparams.dt_init,
                        'dt_scale'  : self.hparams.dt_scale,
                        'dt_init_floor' : self.hparams.dt_init_floor,
                        'conv_bias' : self.hparams.conv_bias,
                        'bias'      : self.hparams.bias,
                        'use_fast_path' : self.hparams.use_fast_path,  # Fused kernel options
                        },
                    rms_norm=False,
                    residual_in_fp32=False,
                    fused_add_norm=False,
                ),
                device=self.hparams.device,
                dtype=torch.float32,
            )
        
        self.fc_decoder_out = torch.nn.Linear(self.hparams.nHiddenFeatures, self.hparams.nClasses + 1)

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

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs['val_loss']).mean()
        avg_acc = sum(self.validation_step_outputs['val_cer']) / len(self.validation_step_outputs['val_cer'])

        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_cer', avg_acc)

        self.validation_step_outputs['val_loss'].clear()
        self.validation_step_outputs['val_cer'].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
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
