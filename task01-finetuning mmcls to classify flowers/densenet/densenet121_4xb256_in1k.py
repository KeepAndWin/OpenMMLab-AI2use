model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='121'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=5,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        ann_file='data/flower/train.txt',
        data_prefix='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        #data_root='data',
        ann_file='data/flower/val.txt',
        data_prefix='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1,))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        #data_root='data',
        ann_file='data/flower/val.txt',
        data_prefix='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1,))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = './densenet121_4xb256_in1k_20220426-07450f99.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
