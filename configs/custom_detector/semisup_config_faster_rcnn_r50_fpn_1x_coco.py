# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_detection import *
    from .._base_.default_runtime import *
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *

experiment_name = 'ALCV'

al_cycle=0
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=False)

json_writer = dict(type='JsonWriter',initial_labeled_size=2500,labeled_size=1000,al_cycle=al_cycle)

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.0005, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=500,
        end=2000,
        by_epoch=False,
        milestones=[1000, 1500],
        gamma=0.1)
]

auto_scale_lr = dict(enable=False, base_batch_size=2)

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=500, val_interval=250)

# val_cfg = None # dict(__delete__=True)
# val_evaluator = None
# val_dataloader = None

test_cfg = None #dict(__delete__=True)
test_evaluator = None
test_dataloader = None

model.data_preprocessor = dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=model.data_preprocessor)

model = dict(
    type='MixedModel',
    model=model,
    data_preprocessor=model.data_preprocessor)

mixed_ann_fold = ""

branch_field = ['sup','unsup_weak','unsup_strong']

labeled_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
    )
]


# pipeline used to augment unlabeled data weakly (doesn't really augment at all
weak_pipeline = [
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
        ]

# pipeline used to augment unlabeled data strongly (not really strong)
strong_pipeline = [
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='RandomFlip', prob=1.0),
                    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

# pipeline used to augment unlabeled data into different views
unlabeled_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_weak=weak_pipeline,
        unsup_strong=strong_pipeline,
    )
]

active_learning_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

active_learning_dataset = dict(
                        type=dataset_type,
                        data_root=data_root,
                        ann_file=mixed_ann_fold+"/unlabeled.json",
                        data_prefix=dict(img='train2017/'),  # makes it such that img is data_root + data_prefix + img name
                        filter_cfg=dict(filter_empty_gt=False),
                        pipeline=active_learning_pipeline,
                        backend_args=backend_args
                        )

labeled_dataset = dict(
                                type=dataset_type,
                                data_root=data_root,
                                ann_file=mixed_ann_fold+"/labeled.json",
                                data_prefix=dict(img='train2017/'), # makes it such that img is data_root + data_prefix + img name
                                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                                pipeline=labeled_pipeline,
                                backend_args=backend_args
                    )

unlabeled_dataset = dict(
                        type=dataset_type,
                        data_root=data_root,
                        ann_file=mixed_ann_fold+"/unlabeled.json",
                        data_prefix=dict(img='train2017/'),  # makes it such that img is data_root + data_prefix + img name
                        filter_cfg=dict(filter_empty_gt=False),
                        pipeline=unlabeled_pipeline,
                        backend_args=backend_args
                        )


unlabeled_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=active_learning_dataset)


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=4,
        source_ratio=[1, 1]),
    dataset=dict(
        type='MixedDataset', labeled=labeled_dataset, unlabeled=unlabeled_dataset))

# custom_hooks = [dict(type='CreateJsonFiles',labeled_size=1000,al_cycle=al_cycle)]