"""This module is used to implement and register the custom datasets.

If OpenMMLab series repositries have supported the target dataset, for example,
CocoDataset. You can simply use it by setting ``type=mmdet.CocoDataset`` in the
config file.

If you want to do some small modifications to the existing dataset,
you can inherit from it and override its methods:

Examples:
    >>> from mmdet.datasets import CocoDataset as MMDetCocoDataset
    >>>
    >>> class CocoDataset(MMDetCocoDataset):
    >>>     def load_data_list(self):
    >>>         ...

Don't worry about the duplicated name of the custom ``CocoDataset`` and the
mmdet ``CocoDataset``, they are registered into different registry nodes.

The default implementation only does the register process. Users need to rename
the ``CustomDataset`` to the real name of the target dataset, for example,
``WiderFaceDataset``, and then implement it.
"""

from mmengine.dataset import BaseDataset
from mmengine_template.registry import DATASETS
from mmdet.datasets import ConcatDataset #, build_dataset


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    ...



@DATASETS.register_module()
class MixedDataset(ConcatDataset):
    """Wrapper for full+partial supervision od."""

    def __init__(self, labeled: dict, unlabeled: dict, **kwargs):
        ignore_keys = ["full", "mixed"]
        for k in ignore_keys:
            if kwargs.get(k) is not None:
                kwargs.pop(k)
        super().__init__([DATASETS.build(labeled), DATASETS.build(unlabeled)], **kwargs)

    @property
    def labeled(self):
        return self.datasets[0]

    @property
    def unlabeled(self):
        return self.datasets[1]