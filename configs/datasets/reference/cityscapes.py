data_root = 'data/Cityscapes/'

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

reference_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CityscapesDataset',
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=data_root,
        pipeline=pipeline,
        ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

