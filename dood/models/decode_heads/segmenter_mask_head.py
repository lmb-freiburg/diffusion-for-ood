import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.decode_heads.segmenter_mask_head import SegmenterMaskTransformerHead as SMTH

del MODELS.module_dict['SegmenterMaskTransformerHead']


@MODELS.register_module()
class SegmenterMaskTransformerHead(SMTH):
    def __init__(self, *args, **kwargs):
        super(SegmenterMaskTransformerHead, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches = x[:, :-self.num_classes] @ self.patch_proj.weight
        cls_seg_feat = x[:, -self.num_classes:] @ self.classes_proj.weight

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)
        
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return masks