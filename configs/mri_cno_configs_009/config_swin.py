_base_ = ['../swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py']
dataset_type = 'CustomDataset'
data_root = '/mnt/g/Programming/mri_cno_data/'
img_norm_cfg = dict(
    mean=[34.94595616, 34.94595616, 34.94595616], std=[22.97774684, 22.97774684, 22.97774684], to_rgb=False)
crop_size = (256, 464)
# crop_size = (256, 256)
classes = ['background', 'focus']
model = dict(
    decode_head=dict(num_classes=len(classes),
                     #  ignore_index=0,
                     loss_decode=[
        dict(type='FocalLoss',
             loss_name='loss_focal',
             loss_weight=1.0,
             class_weight=[0.1, 0.9]
             ),
        dict(type='DiceLoss',
             loss_name='loss_dice',
             loss_weight=3.0,
             class_weight=[0.1, 0.9]
             ),
    ],
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=80000),
    ),
    auxiliary_head=dict(num_classes=len(classes),
                        # ignore_index=0,
                        loss_decode=[
        dict(type='FocalLoss',
             loss_name='loss_focal',
             loss_weight=1.0,
             class_weight=[0.1, 0.9]
             ),
             dict(type='DiceLoss',
             loss_name='loss_dice',
             loss_weight=3.0,
             class_weight=[0.1, 0.9]
             ),
    ]
    ),
    test_cfg=dict(mode='whole')
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=16, pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='Pad', size_divisor=16),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        # _delete_=True,
        type=dataset_type,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        ignore_index=-1,
        pipeline=train_pipeline,
        data_root='/mnt/g/Programming/cno_raw_data/',
        img_dir='selected_images/train/',
        ann_dir='selected_masks/train_01_1_class/',
        # data_root='/mnt/g/Programming/brain_mri_segmentation/',
        # img_dir='train_images_png',
        # ann_dir='train_masks_01_png',


    ),
    val=dict(
        type=dataset_type,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        ignore_index=-1,
        data_root='/mnt/g/Programming/cno_raw_data/',
        img_dir='selected_images/val/',
        ann_dir='selected_masks/val_01_1_class/',
        # data_root='/mnt/g/Programming/brain_mri_segmentation/',
        # img_dir='val_images_png',
        # ann_dir='val_masks_01_png',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        img_suffix='.png',
        classes=classes,
        ignore_index=-1,
        data_root='/mnt/g/Programming/cno_raw_data/',
        img_dir='selected_images/val/',
        ann_dir='selected_masks/val_01_1_class/',
        pipeline=test_pipeline
    )
)
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mDice', pre_eval=True,
                  # ignore_index=0
                  )
gpu_ids = [0]

auto_resume = False
