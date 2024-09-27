from mmseg.models.backbones.vit import VisionTransformer
from torch import nn
import torch
from torch.nn import functional as F
import math

from mmengine.model.weight_init import trunc_normal_
from mmseg.registry import MODELS


@MODELS.register_module()
class DistilledVisionTransformer(VisionTransformer):
    num_extra_tokens = 2  # cls_token, dist_token

    def __init__(self, *args, patch_bias=True, **kwargs):
        super(DistilledVisionTransformer, self).__init__(*args, patch_bias=patch_bias, **kwargs)
        self.embed_dims = self.cls_token.shape[-1]
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.patch_resolution = (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size)
        pos_embed_shape = self.pos_embed.shape
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_shape[1]+1, pos_embed_shape[2]))

    @staticmethod
    def resize_pos_embed(posemb, grid_new_shape, grid_old_shape, mode, num_extra_tokens=2):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        posemb_tok, posemb_grid = (
            posemb[:, :num_extra_tokens],
            posemb[0, num_extra_tokens:],
        )
        if grid_old_shape is None:
            gs_old_h = int(math.sqrt(len(posemb_grid)))
            gs_old_w = gs_old_h
        else:
            gs_old_h, gs_old_w = grid_old_shape

        gs_h, gs_w = grid_new_shape
        embed_dim = posemb_grid.shape[-1]
        posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, embed_dim).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode=mode)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, embed_dim)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        # x = self._pos_embeding(x, hw_shape, self.pos_embed)
        pos_embed = self.resize_pos_embed(
            self.pos_embed,
            hw_shape,
            self.patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = x + pos_embed
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 2:]
                else:
                    raise NotImplementedError
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    raise NotImplementedError
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def init_weights(self):
        super(DistilledVisionTransformer, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.dist_token, std=0.02)

