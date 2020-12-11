norm_cfg = dict(type='MMSyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        stem_channels=64,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='MMSyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        # contract_dilation=False),
        contract_dilation=True),
    decode_head=dict(
        type='ASPPDecoupleSegHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        edge_attention_thresh=0.8,
        loss_body_decode=dict(
            type='SoftCrossEntropyLoss',
            img_based_class_weights='no_norm',
            batch_weights=False,
            customsoftmax=True,
            loss_weight=1.0),
        loss_edge_decode=dict(
            type='CrossEntropyLoss',
            img_based_class_weights='norm',
            batch_weights=True,
            loss_weight=20.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='MMSyncBN', requires_grad=True),
        align_corners=False,
        # align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='MMSyncBN', requires_grad=True),
        align_corners=False,
        # align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(832, 832), stride=(554, 554))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(832, 832), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(832, 832), pad_val=0, seg_pad_val=255),
            dict(
                type='GenerateEdgeMap',
                num_classes=19,
                radius=2,
                ignore_index=255),
            dict(
                type='BoundaryRelaxedOneHot',
                num_classes=19,
                border_window=1,
                strict_border_classes=None,
                ignore_index=255),
            dict(type='DefaultFormatBundle'),
            # dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            dict(
                type='Collect',
                keys=dict(
                    img='img',
                    gt_semantic_seg=[
                        'gt_semantic_seg', 'boundary_relaxed_one_hot',
                        'edge_map'
                    ]))
        ]),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                # img_scale=(2049, 1025),
                img_scale=(2048, 1024),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                img_ratios=[
                    1.0,
                ],
                # flip=True,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/deeplabv3plus_dsn_paper/latest.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
total_iters = 80000
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=3)
evaluation = dict(interval=8000, metric='mIoU')
# find_unused_parameters=True
