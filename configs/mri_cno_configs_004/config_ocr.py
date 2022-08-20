_base_ = ['../ocrnet/ocrnet_hr48_512x1024_80k_cityscapes.py']
dataset_type = 'CustomDataset'
data_root = '/mnt/g/Programming/mri_cno_data/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
crop_size = (256, 464)
# crop_size = (256, 256)
classes = ['background', 'focus']
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=len(classes),
            align_corners=False,
            loss_decode=[
                dict(type='DiceLoss',
                     loss_name='loss_dice',
                     loss_weight=3.0,
                     class_weight=[0.1, 0.9]
                     ),
                dict(type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=[0.1, 0.9]
                     )
            ]
        ),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=len(classes),
            align_corners=False,
            loss_decode=[
                dict(type='DiceLoss',
                     loss_name='loss_dice',
                     loss_weight=3.0,
                     class_weight=[0.1, 0.9]
                     ),
                dict(type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=[0.1, 0.9]
                     )
            ]
        )
    ])

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
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mDice', pre_eval=True,
                  # ignore_index=0
                  )

gpu_ids = [0]

auto_resume = False
