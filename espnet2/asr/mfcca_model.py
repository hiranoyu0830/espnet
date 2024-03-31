from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)
import random
import math
import numpy as np
import pdb
import scipy

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class MFCCACTCModel(ESPnetASRModel):
    '''e2e mfcca model'''
    
    def __init__(
            self,
            vocab_size: int,
            token_list: Union[Tuple[str, ...], List[str]],
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            label_aggregator: Optional[torch.nn.Module],
            preencoder: Optional[AbsPreEncoder],
            encoder: AbsEncoder,
            postencoder: Optional[AbsPostEncoder],
            decoder: AbsDecoder,
            ctc: CTC,
            joint_network: Optional[torch.nn.Module] = None,
            ctc_weight: float = 0.0,
            interce_weight: float = 0.0,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            mask_ratio: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = False,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            ctc_weight=ctc_weight,
            #interce_weight=interce_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )
        self.interce_weight = interce_weight
        self.mask_ratio = mask_ratio
        self.label_aggregator = label_aggregator
        self.max_spk_num = 4
        self.power_weight = torch.from_numpy(2 ** np.arange(self.max_spk_num)[np.newaxis, np.newaxis, :]).float()
        #self.int_token_arr = torch.from_numpy(np.array(self.token_list).astype(int)[np.newaaxis, np.newaxis, :]).int()
        # Define conditioning layer
        # 16 is from 2 ** 4(maximum number of spk)
        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            # spk -> feature
            # used only for conditioning
            self.encoder.conditioning_layer = torch.nn.Linear(
                16, self.encoder.output_size()
            )
            
        self.error_calculator = None
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )


    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        spk_labels: torch.Tensor,
        spk_labels_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        assert speech.size(2) == 8, speech.shape
        #print(f"channel size: {speech.size(2)}")
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        if (speech.dim() == 3 and speech.size(2) == 8 and self.mask_ratio != 0):
            rate_num = random.random()
            # rate_num = 0.1
            if (rate_num <= self.mask_ratio):
                retain_channel = math.ceil(random.random() * 8)
                if (retain_channel > 1):
                    speech = speech[:, :, torch.randperm(8)[0:retain_channel].sort().values]
                else:
                    speech = speech[:, :, torch.randperm(8)[0]]
        #pdb.set_trace()
        batch_size = speech.shape[0]
        # for data-parallel
        text = text[:, : text_lengths.max()]
        stats = dict()

        # Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )      
        #spk_labels, spk_labels_lengths = spk_labels.to(speech.device), spk_labels_lengths.to(speech.device)
        # Aggregate labels after conv subsampling
        # Do majority vote twice to make sure the labels are consistent
        #print("before majority vote: ", spk_labels)
        spk_labels, spk_labels_lengths = self.fast_majority_vote(
            spk_labels, spk_labels_lengths
        )
        #print("after majority vote:1 ", spk_labels.shape, spk_labels_lengths.shape)
        spk_labels, spk_labels_lengths = self.fast_majority_vote(
            spk_labels, spk_labels_lengths
        )
        #print("after majority vote:2 ", spk_labels)

        #pdb.set_trace()
        # 1. Calculate power-set encoding (PSE) labels
        #pad_bin_labels = F.pad(spk_labels, (0, self.max_spk_num - spk_labels.shape[2]), "constant", 0.0)
        #raw_pse_labels = torch.sum(pad_bin_labels * self.power_weight, dim=2, keepdim=True)
        #print("spk_labels_device: ", spk_labels.device)
        #print("power_weight_device: ", self.power_weight.device)   
        #spk_labels = self.spk_label_sorter(spk_labels)
        #print("sorted_spk_labels: ", spk_labels)
        power_weight = self.power_weight.to(spk_labels.device)
        pse_labels = torch.sum(spk_labels * power_weight, dim=2, keepdim=True).float()
        #print("pse_labels: ", pse_labels)
        #pse_labels = torch.argmax((raw_pse_labels.int() == self.int_token_arr).float(), dim=2)
        spk_labels = spk_labels.long()
        pse_labels = pse_labels.long()

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]
            #print("encoder_out: ", encoder_out.shape)

        # 2a. Attention-decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None

        # 2c. Inter-CE (optional)
        loss_interce = 0.0
        for layer_idx, intermediate_out in intermediate_outs:

            # intermediate_outs have not been fed into linear layer yet
            #print("intermediate_out: ", intermediate_out.device)
            #print("pse_labels: ", pse_labels.device)
            #print("batch_size: ", batch_size)
            loss_ce = self.encoder.ce_loss(intermediate_out, pse_labels)
            loss_interce = loss_interce + loss_ce

            # Collect Intermedaite ce stats
            stats["loss_interce_layer{}".format(layer_idx)] = (
                loss_ce.detach() if loss_ce is not None else None
            )

        #loss_interce = loss_interce / len(intermediate_outs)
        # Collect the mean of inter-ce stats
        stats["loss_interce"] = loss_interce.detach() if loss_interce is not None else None

        # 3. Joint-network
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + self.interce_weight * loss_interce

        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["ctc_cer"] = cer_ctc

        stats["loss"] = loss.detach() if loss is not None else None

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        spk_labels: torch.Tensor,
        spk_labels_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths, channel_size = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths, channel_size = self._extract_feats(speech, speech_lengths)
            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)
        # pdb.set_trace()

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning or getattr(
            self.encoder, "ctc_trim", False
        ):
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, channel_size, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, channel_size)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (encoder_out.dim() == 4):
            assert encoder_out.size(2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )
        else:
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )
        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths, channel_size = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
            channel_size = 1
        return feats, feats_lengths, channel_size

    
    def majority_vote(
        self, labels: torch.Tensor, labels_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Majority vote for time-domain labels
        Args:
            labels: (Batch, Length, spk_num)
            labels_lengths: (Batch, )
        """
        #print("labels", labels)
        kernel_size = 3 # should be same as the kernel size of the conv layer
        stride = 2 # should be same as the kernel size of the conv layer

        # Calculate the number of windows based on the stride
        #num_windows = (labels_lengths.item() - kernel_size) // stride + 1
        n_spk = labels.size(2)
        n_batch = labels.size(0)
        assert n_spk == 4

        labels_subsampled = []
        for i in range(n_batch):
            spk_label = labels[i]
            spk_label = spk_label.t()

            num_windows = (labels.size(1)- kernel_size) // stride + 1
            label_subsampled = []

            for spk in range(n_spk):
                labels_subsampled_spk = []

                for j in range(num_windows):
                    start = j * stride
                    end = start + kernel_size
                    window_labels = spk_label[spk][start:end]
                    #print("window_labels", window_labels)
                    majority = torch.mode(window_labels, dim=0).values.to(labels.device)
                    #print(majority)
                    # Replace the original labels with the majority labels
                    #labels[:, start:end, :] = majority_labels.unsqueeze(1).expand(-1, kernel_size, -1)
                    labels_subsampled_spk.append(majority) 

                label_subsampled.append(labels_subsampled_spk)

            label_subsampled = torch.Tensor(label_subsampled).to(labels.device)
            #print("label_subsampled", label_subsampled)
            label_subsampled = label_subsampled.t()
            labels_subsampled.append(label_subsampled)

        labels_subsampled = torch.stack(labels_subsampled)
        labels_subsampled = labels_subsampled.to(labels.device)
        return labels_subsampled, labels_lengths


    def fast_majority_vote(self, labels: torch.Tensor, labels_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Majority vote for time-domain labels
        Args:
            labels: (Batch, Length, spk_num)
            labels_lengths: (Batch, )
        """
        kernel_size = 3  # should be same as the kernel size of the conv layer
        stride = 2  # should be same as the kernel size of the conv layer

        # Unfold the labels tensor along the time dimension to create sliding windows
        windows = labels.unfold(dimension=1, size=kernel_size, step=stride)

        # Compute the mode along the time dimension for each window and each speaker
        majority, _ = torch.mode(windows, dim=3)

        return majority, labels_lengths
    
    def spk_label_sorter(self, labels: torch.Tensor) -> torch.Tensor:
        """Sort the speaker labels in starting time order
        Args:
            labels: (Batch, Length, spk_num)
        """
        labels = labels.unfold(dimension=1, size=labels.shape[1], step=1)
        # labels: (Batch, 1, spk_num, length)
        labels = labels.squeeze(1)
        # labels: (Batch, spk_num, length)
        # Find the first active time step for each speaker
        first_indices = []

        first_active = torch.nonzero(labels)

        # Get the indices that would sort the speakers by their first active time step
        sorted_indices = torch.argsort(first_active[2])

        # Sort the activity tensor by these indices
        sorted_labels = labels[:, :, sorted_indices]

        return sorted_labels