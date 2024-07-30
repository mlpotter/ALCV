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

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)

model.data_preprocessor = dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=model.data_preprocessor)

model = dict(
    type='MixedModel',
    model=model
)

mixed_ann_fold = "/home/mlpotter/Documents/Northeastern/Research/ALCV/work_dirs/custom_config_faster_rcnn_r50_fpn_1x_coco/step_0"

labeled_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=ExtraAttrs,tag="labeled"), # MLP ADD TO TEST
    dict(type=PackDetInputs, meta_keys=['img_id','img_path','ori_shape','img_shape','tag'])
]

unlabeled_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=ExtraAttrs,tag="unlabeled"), # MLP ADD TO TEST
    dict(type=PackDetInputs, meta_keys=['img_id','img_path','ori_shape','img_shape','tag'])
]

mixed_dataset = dict(type='MixedDataset',
                     labeled=dict(
                                type=dataset_type,
                                data_root=data_root,
                                ann_file=mixed_ann_fold+"/labeled.json",
                                data_prefix=dict(img='train2017/'), # makes it such that img is data_root + data_prefix + img name
                                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                                pipeline=labeled_pipeline,
                                backend_args=backend_args),
                     unlabeled=dict(
                        type=dataset_type,
                        data_root=data_root,
                        ann_file=mixed_ann_fold+"/unlabeled.json",
                        data_prefix=dict(img='train2017/'),  # makes it such that img is data_root + data_prefix + img name
                        filter_cfg=dict(filter_empty_gt=True, min_size=32),
                        pipeline=unlabeled_pipeline,
                        backend_args=backend_args)
                    )


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=mixed_dataset)