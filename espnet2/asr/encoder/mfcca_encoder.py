# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet2.asr.encoder.encoder_layer_mfcca import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import (
    get_activation,
    make_pad_mask,
    #trim_by_ctc_posterior,
)
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from espnet2.asr.mfcca_model import MFCCACTCModel
import math
import pdb

class MFCCAEncoder(AbsEncoder):
    """MFCCA Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interce_layer_idx: List[int] = [],
        interce_use_conditioning: bool = False,
        ctc_trim: bool = False,
        #stochastic_depth_rate: Union[float, List[float]] = 0.0,
        #layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        encoder_selfattn_layer_raw = MultiHeadedAttention # for Cross-channel attention
        encoder_selfattn_layer_args_raw = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        """
        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )
        """
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer_raw(*encoder_selfattn_layer_args_raw),
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                #stochastic_depth_rate[lnum],
            ),
            #layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)
        
        self.interce_layer_idx = interce_layer_idx
        if len(interce_layer_idx) > 0:
            assert 0 < min(interce_layer_idx) and max(interce_layer_idx) < num_blocks
        self.interce_use_conditioning = interce_use_conditioning
        if self.interce_use_conditioning:
            self.conditioning_layer = torch.nn.Linear(16, output_size)
        else:
            self.conditioning_layer = None
        #self.conditioning_layer = None
        self.ctc_trim = ctc_trim
        """
        self.conv1 = torch.nn.Conv2d(8, 16, [5, 7], stride=[1, 1], padding=(2, 3))

        self.conv2 = torch.nn.Conv2d(16, 32, [5, 7], stride=[1, 1], padding=(2, 3))

        self.conv3 = torch.nn.Conv2d(32, 16, [5, 7], stride=[1, 1], padding=(2, 3))

        self.conv4 = torch.nn.Conv2d(16, 1, [5, 7], stride=[1, 1], padding=(2, 3))

        """
        self.conv1 = torch.nn.Conv2d(8, 16, [5, 5], stride=[1, 1], padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(16, 8, [5, 5], stride=[1, 1], padding=(2, 2))
        self.conv3 = torch.nn.Conv2d(8, 4, [5, 5], stride=[1, 1], padding=(2, 2))
        self.conv4 = torch.nn.Conv2d(4, 2, [5, 5], stride=[1, 1], padding=(2, 2))
        self.conv5 = torch.nn.Conv2d(2, 1, [5, 5], stride=[1, 1], padding=(2, 2))
        

        # used only for calculating CE loss
        self.inter_linear = torch.nn.Linear(output_size, 16)
        self.ce = torch.nn.CrossEntropyLoss()


    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        channel_size: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): ctc module for intermediate CTC loss
            return_all_hs (bool): whether to return all hidden states

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        #pdb.set_trace()
        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        #xs_pad, masks, channel_size = self.encoders(xs_pad, masks, channel_size)
        #if isinstance(xs_pad, tuple):
            #xs_pad = xs_pad[0]

        #print("shape of after subsampling: ", xs_pad.size())
        intermediate_outs = []
        if len(self.interce_layer_idx) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks, _ = encoder_layer(xs_pad, masks, channel_size)
                if return_all_hs:
                    if isinstance(xs_pad, tuple):
                        intermediate_outs.append(xs_pad[0])
                    else:
                        intermediate_outs.append(xs_pad)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks_inter, channel_size_inter = encoder_layer(xs_pad, masks, channel_size)
                
                if layer_idx + 1 in self.interce_layer_idx:
                    encoder_out = xs_pad
                    
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]
                    #print("inter_encoder_out: ", encoder_out.shape)
                    t_leng = encoder_out.size(1)
                    d_dim = encoder_out.size(2)
                    encoder_out = encoder_out.reshape(-1, channel_size_inter, t_leng, d_dim)
                    #print("inter_encoder_out: ", encoder_out.shape)
                    if (channel_size_inter < 8):
                        repeat_num = math.ceil(8 / channel_size_inter) # 小数点以下切り上げ
                        encoder_out = encoder_out.repeat(1, repeat_num, 1, 1)[:, 0:8, :, :]
                    #print("inter_encoder_out_after_repeat: ", encoder_out.shape)
                    # Perform channel fusion
                    # Channel_size: 8(channel_size after channel masking) -> 1
                    encoder_out = self.conv1(encoder_out)
                    encoder_out = self.conv2(encoder_out)
                    encoder_out = self.conv3(encoder_out)
                    encoder_out = self.conv4(encoder_out)
                    encoder_out = self.conv5(encoder_out)
                    #print("inter_encoder_out: ", encoder_out.shape)

                    encoder_out = encoder_out.squeeze().reshape(-1, t_leng, d_dim) # T x D
                    mask_tmp = masks_inter.size(1)
                    masks_inter = masks_inter.reshape(-1, channel_size_inter, mask_tmp, t_leng)[:, 0, :, :]

                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)
                    # This is for intermediate loss calculation
                    #print("intermediate_outs: ", encoder_out.shape)
                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    # Apply self-conditing
                    if self.interce_use_conditioning:
                        # This softmax function contains a linear layer
                        ce_out = self._softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            cond = self.conditioning_layer(ce_out)
                            cond_mc = cond.repeat(1, channel_size_inter, 1, 1)
                            cond_mc = cond_mc.squeeze().reshape(-1, t_leng, d_dim) # T x D
                            #print("cond_mc: ", cond_mc.shape)
                            #print("x: ", x.shape)
                            assert cond_mc.size() == x.size()
                            x = x + cond_mc
                            xs_pad = (x, pos_emb)
                        else:
                            cond = self.conditioning_layer(ce_out)
                            cond_mc = cond.repeat(1, channel_size, 1, 1)                        
                            xs_pad = xs_pad + cond_mc

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        t_leng = xs_pad.size(1)
        d_dim = xs_pad.size(2)
        xs_pad = xs_pad.reshape(-1, channel_size, t_leng, d_dim)

        if (channel_size < 8):
            repeat_num = math.ceil(8 / channel_size) # 小数点以下切り上げ
            xs_pad = xs_pad.repeat(1, repeat_num, 1, 1)[:, 0:8, :, :]
        xs_pad = self.conv1(xs_pad)
        xs_pad = self.conv2(xs_pad)
        xs_pad = self.conv3(xs_pad)
        xs_pad = self.conv4(xs_pad)
        xs_pad = self.conv5(xs_pad)
        xs_pad = xs_pad.squeeze().reshape(-1, t_leng, d_dim) # T x D
        mask_tmp = masks.size(1)
        masks = masks.reshape(-1, channel_size, mask_tmp, t_leng)[:, 0, :, :]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

    def forward_hidden(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.
        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        if (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
        num_layer = len(self.encoders)
        for idx, encoder in enumerate(self.encoders):
            xs_pad, masks = encoder(xs_pad, masks)
            if idx == num_layer // 2 - 1:
                hidden_feature = xs_pad
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
            hidden_feature = hidden_feature[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
            self.hidden_feature = self.after_norm(hidden_feature)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
    
    def ce_loss(self, encoder_out, ys_pad):
        """Calculate cross entropy loss.
        Args:
            encoder_out (torch.Tensor): Output from forward function.
            ys_pad (torch.Tensor): Target token ids padded with 0.
        Returns:
            loss (torch.Tensor): A tensor holding a scalar loss value.
        """

        encoder_out = self.inter_linear(encoder_out) # 256 -> 16
        encoder_out = encoder_out.view(-1, encoder_out.shape[-1])
        ys_pad = ys_pad.view(-1)
        #print("after_linear: ", encoder_out.shape)
        loss = self.ce(encoder_out, ys_pad)
        return loss
    
    def _softmax(self, encoder_out: torch.Tensor):
        return F.softmax(self.inter_linear(encoder_out), dim=2)
