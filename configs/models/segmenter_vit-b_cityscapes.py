_base_ = "../datasets/reference/cityscapes.py"

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    pad_val=0,
    seg_pad_val=255,
    size=(512, 1024,),
    std=[58.395, 57.12, 57.375],
    type='SegDataPreProcessor')

model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DistilledVisionTransformer',
        img_size=(768, 768),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bicubic',
        init_cfg=None),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=19,
        num_layers=1,
        num_heads=6,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole'),
)

custom_hooks = [
    dict(
        type='DiffusionSetupHook',
        priority='VERY_HIGH',
        ft_size=768,
        diffusion_denoiser_channels=768,
        num_steps_in_parallel=25
        ),
]

