_base_ = "../datasets/reference/ade20k.py"

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    std=[58.395, 57.12, 57.375],
    type='SegDataPreProcessor')

model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=True,
        patch_bias=True,
        interpolate_mode='bilinear',
        init_cfg=None),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=1024,
        channels=256,
        in_index=3,
        num_classes=150,
        dropout_ratio=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_convs=1,
        up_scale=4,
        kernel_size=1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='whole')
)


custom_hooks = [
    dict(
        type='DiffusionSetupHook',
        priority='VERY_HIGH',
        ft_size=1024,
        diffusion_denoiser_channels=1024,
        diffusion_model_channel_mult=(1, 1, 1, 1, 1, 1, 1, 1),
        diffusion_model_num_middle_blocks=4,
        normalization_queue_size=4096,
        num_steps_in_parallel=25,
        diffusion_model_time_embed_mode='scalar+identity',
        ),
]