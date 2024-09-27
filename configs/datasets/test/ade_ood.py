data_root = 'data/ADE-OoD'

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='BaseOoDDataset',
        ood_index=1,
        data_prefix=dict(img_path='images', seg_map_path='annotations'),
        img_suffix='.jpg',
        seg_map_suffix='_mask.png',
        data_root=data_root,
        pipeline=pipeline,
        ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))