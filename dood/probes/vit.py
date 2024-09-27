from .utils import MultiHeadAttentionExplicitQKV
from .base import FeatureProbe

class FeatureProbeViT(FeatureProbe):
    @staticmethod
    def _register_hooks(model, layer_index=-1, eval_all_features=False):
        model.backbone.layers[layer_index].attn.attn = MultiHeadAttentionExplicitQKV(
            model.backbone.layers[layer_index].attn.attn)
        feat_dict = dict()

        # this may not work in a few special cases
        if hasattr(model.backbone, "num_extra_tokens"):
            num_extra_tokens = model.backbone.num_extra_tokens
        elif model.backbone.with_cls_token:
            num_extra_tokens = 1
        else:
            num_extra_tokens = 0
        
        FeatureProbe._register_activation_hook(model.backbone.patch_embed,
                                               lambda x: x[1],  # x[0] is the fwd pass tensor, we only need the shape
                                               feat_dict,
                                               "hw_shape")
        if eval_all_features:
            FeatureProbe._register_activation_hook(model.backbone.layers[layer_index].attn.attn.explicit_qkv,
                                                   lambda ft: [ft_[num_extra_tokens:] for ft_ in ft],
                                                   feat_dict,
                                                   "qkv")
            FeatureProbe._register_activation_hook(model.backbone.layers[layer_index],
                                                   lambda ft: ft[0, num_extra_tokens:],
                                                   feat_dict,
                                                   "block_output")
        else:
            # only keys
            FeatureProbe._register_activation_hook(model.backbone.layers[layer_index].attn.attn.explicit_qkv,
                                                   lambda ft: ft[1][num_extra_tokens:],
                                                   feat_dict,
                                                   "k")
        return feat_dict

    @staticmethod
    def _features_post_process(ft_dict):
        for k, ft in zip("qkv", ft_dict.pop("qkv", [])):
            ft_dict[k] = ft
        hw_shape = ft_dict.pop("hw_shape")
        return {k: ft.reshape(*hw_shape, ft.shape[-1]) for k, ft in ft_dict.items()}



        