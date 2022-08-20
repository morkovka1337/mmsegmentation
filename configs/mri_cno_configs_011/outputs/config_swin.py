norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
        )),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                loss_weight=1.0,
                class_weight=[0.01, 0.99]),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=3.0,
                class_weight=[0.01, 0.99])
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                loss_weight=1.0,
                class_weight=[0.01, 0.99]),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=3.0,
                class_weight=[0.01, 0.99])
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CustomDataset'
data_root = '/mnt/g/Programming/mri_cno_data/'
img_norm_cfg = dict(
    mean=[34.94595616, 34.94595616, 34.94595616],
    std=[22.97774684, 22.97774684, 22.97774684],
    to_rgb=False)
crop_size = (256, 464)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 464), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 464), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[34.94595616, 34.94595616, 34.94595616],
        std=[22.97774684, 22.97774684, 22.97774684],
        to_rgb=False),
    dict(type='Pad', size_divisor=16, pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 464),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[34.94595616, 34.94595616, 34.94595616],
                std=[22.97774684, 22.97774684, 22.97774684],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        data_root='/mnt/g/Programming/cno_raw_data/',
        img_dir='selected_images/train/',
        ann_dir='selected_masks/train_01_1_class/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 464), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 464), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[34.94595616, 34.94595616, 34.94595616],
                std=[22.97774684, 22.97774684, 22.97774684],
                to_rgb=False),
            dict(type='Pad', size_divisor=16, pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['background', 'focus'],
        ignore_index=-1),
    val=dict(
        type='CustomDataset',
        data_root='/mnt/g/Programming/cno_raw_data/',
        img_dir='selected_images/val/',
        ann_dir='selected_masks/val_01_1_class/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 464),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[34.94595616, 34.94595616, 34.94595616],
                        std=[22.97774684, 22.97774684, 22.97774684],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['background', 'focus'],
        ignore_index=-1),
    test=dict(
        type='CustomDataset',
        data_root='/mnt/g/Programming/cno_raw_data/',
        img_dir='selected_images/val/',
        ann_dir='selected_masks/val_01_1_class/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 464),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[34.94595616, 34.94595616, 34.94595616],
                        std=[22.97774684, 22.97774684, 22.97774684],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        img_suffix='.png',
        classes=['background', 'focus'],
        ignore_index=-1))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mDice', pre_eval=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
classes = ['background', 'focus']
gpu_ids = [0]
auto_resume = False
work_dir = 'configs/mri_cno_configs_011/outputs'
