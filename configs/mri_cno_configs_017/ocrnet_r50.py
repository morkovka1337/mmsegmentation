# base_ = ['../deeplabv3/deeplabv3_r50-d8_512x512_80k_ade20k.py']
dataset_type = 'CustomDataset'
data_root = '/mnt/g/Programming/mri_cno_data/'
img_norm_cfg = dict(
    mean=[34.94595616, 34.94595616, 34.94595616], std=[22.97774684, 22.97774684, 22.97774684], to_rgb=False)
# crop_size = (256, 256)
crop_size = (256, 320)
classes = ['background', 'focus']
class_weights = [0.2, 0.8]
norm_cfg = dict(type='SyncBN',
                requires_grad=True
                )
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=len(classes),
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(type='FocalLoss',
                     loss_name='loss_focal',
                     loss_weight=1.0,
                     class_weight=class_weights
                     ),
                dict(type='DiceLoss',
                     loss_name='loss_dice',
                     use_sigmoid=True,
                     loss_weight=1.0,
                     class_weight=class_weights
                     ),
            ],
        ),
        dict(
            type='OCRHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=len(classes),
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(type='FocalLoss',
                     loss_name='loss_focal',
                     loss_weight=1.0,
                     class_weight=class_weights
                     ),
                dict(type='DiceLoss',
                     use_sigmoid=True,
                     loss_name='loss_dice',
                     loss_weight=1.0,
                     class_weight=class_weights
                     ),
            ],

        )
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=crop_size,
        # keep_ratio=False,
        # ratio_range=(0.5, 2.0)
    ),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=16, pad_val=0),
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
            dict(type='Resize', keep_ratio=True),
            # dict(type='Pad', size_divisor=16),
            # dict(type='RandomFlip'),
            dict(type='Normalize',
                 **img_norm_cfg
                 ),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
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
        # img_dir='selected_images/val/',
        # ann_dir='selected_masks/val_01_1_class/',
        img_dir='selected_images/train/',
        ann_dir='selected_masks/train_01_1_class/',
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
    interval=10, hooks=[
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
    # _delete_=True,
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    # weight_decay=0.01,
)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=81, metric='mDice', pre_eval=True,
                  # ignore_index=0
                  )
gpu_ids = [0]

auto_resume = False
