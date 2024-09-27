from .vit import FeatureProbeViT

def get_feature_probe(model):
    if model.backbone.__class__.__name__ in ["VisionTransformer", "DistilledVisionTransformer", "SETR"]:
        feature_probe = FeatureProbeViT(model, layer_index=-1)
    else:
        raise NotImplementedError("Backbone type:", model.backbone.__class__.__name__)
    
    return feature_probe
    