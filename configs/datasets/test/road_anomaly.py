data_root='data/RoadAnomaly_jpg/'

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='BaseOoDDataset',
        data_root=data_root,
        ood_index=2,
        data_prefix=dict(img_path='frames', seg_map_path='frames'),
        img_suffix='.jpg',
        seg_map_suffix='.labels/labels_semantic.png',
        pipeline=pipeline,
        ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))