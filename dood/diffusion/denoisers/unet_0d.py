from abc import abstractmethod

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(dtype=torch.float32) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def dumb_timestep_embedding(timesteps, dim):
    """
    :return: an [N x dim] Tensor of positional embeddings.
    """
    return timesteps.unsqueeze(1).expand(-1, dim).to(dtype=torch.float32)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        # self.to(torch.float32)
        # return super().forward(x.to(dtype=torch.float32)).type(x.dtype)
        return super().forward(x)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
        

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels is not None else channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Linear(channels, self.out_channels),
        )
        
        if emb_channels > 0:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    self.out_channels,
                ),
            )
        else:
            self.emb_layers = nn.Identity()
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Linear(self.out_channels, self.out_channels)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(channels, self.out_channels)

    def forward(self, x, t_emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C] Tensor of outputs.
        """
        h = self.in_layers(x)
        h_emb = self.emb_layers(t_emb).type(h.dtype)
        h = h + h_emb
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNet0D(nn.Module):
    """
    UNet model with timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks=1,  # per stage
        channel_mult=(1, 1, 1, 1),  # length indicates number of encoder/decoder stages
        num_middle_blocks=2,
        add_input_conversion_block=False,  # add one block at the start to convert from in_channels to model_channels
        add_dec_conversion_blocks=False,  # add one block in each decoder stage to convert from the skip+upsample channels to the stage channels
        dropout=0,
        time_embed_dim_factor=1,
        timestep_embedding_mode="periodic+embed",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        time_embed_dim = model_channels * time_embed_dim_factor

        # the following is only for ResBlocks. The above is the initial time embedding.
        time_embed_fn, time_embed_mode = timestep_embedding_mode.split("+")

        if time_embed_fn == "periodic":
            self.timestep_embedding_fn = timestep_embedding
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        elif time_embed_fn == "scalar":
            self.timestep_embedding_fn = dumb_timestep_embedding
            self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim))
        else:
            raise ValueError

        if time_embed_mode == "identity":
            time_embed_dim *= -1  # this tells the resblocks to use nn.Identity

        res_block = ResBlock
        
        if add_input_conversion_block:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        nn.Linear(in_channels, model_channels)
                    )
                ]
            )
            input_block_chans = [model_channels]
            ch = model_channels
        else:
            self.input_blocks = nn.ModuleList([])
            input_block_chans = []
            ch = in_channels
            
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    res_block(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

        self.middle_blocks = nn.ModuleList([])
        for _ in range(num_middle_blocks):
            layers = [
                res_block(
                    ch,
                    time_embed_dim,
                    dropout,
                )
            ]
            self.middle_blocks.append(TimestepEmbedSequential(*layers))

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + (1 if add_dec_conversion_blocks else 0)):
                layers = [
                    res_block(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                    )
                ]
                ch = model_channels * mult
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, out_channels)),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        ts_embed = self.timestep_embedding_fn(timesteps, self.model_channels)
        emb = self.time_embed(ts_embed.to(dtype=x.dtype))

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        for module in self.middle_blocks:
            h = module(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)
