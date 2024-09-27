data_root = 'data/RoadAnomaly21/dataset_AnomalyTrack/'

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
        ood_index=1,
        data_prefix=dict(img_path='images_val', seg_map_path='labels_masks'),
        img_suffix='.jpg',
        seg_map_suffix='_labels_semantic.png',
        pipeline=pipeline,
        ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))