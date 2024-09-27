data_root = 'data/fishyscapes_laf/'

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='BaseOoDDataset',
        ood_index=25,
        data_prefix=dict(img_path='images', seg_map_path='annotations'),
        data_root=data_root,
        img_suffix='image.png',
        seg_map_suffix='mask.png',
        pipeline=pipeline,
        ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))