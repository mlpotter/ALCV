import os
import os.path as osp
import glob
import shutil
import json
import numpy as np
import time

def patch_cfg(cfg):

    cfg.work_dir = osp.join(cfg.base_work_dir,f"step_{cfg.al_cycle}")

    cfg.mixed_ann_fold = osp.abspath(cfg.work_dir)

    if cfg.train_dataloader.dataset.get('datasets',False):
        for dataset in cfg.train_dataloader.dataset.datasets:
            dataset.ann_file = osp.join(cfg.mixed_ann_fold,dataset.ann_file)
    else:
        cfg.train_dataloader.dataset.ann_file = osp.join(cfg.mixed_ann_fold,cfg.train_dataloader.dataset.ann_file)

    if cfg.get('val_evaluator',None) is not None:
        cfg.val_evaluator.ann_file = osp.join(cfg.data_root,cfg.data_val_ann_file)
    if cfg.get('test_evaluator',None) is not None:
        cfg.test_evaluator.ann_file = osp.join(cfg.data_root,cfg.data_val_ann_file)

    cfg.active_learning_dataloader.dataset.ann_file = osp.join(cfg.mixed_ann_fold,cfg.active_learning_dataloader.dataset.ann_file)

    # for dataset in cfg.train.
    # cfg.train_dataloader.dataset.labeled.ann_file = osp.join(cfg.mixed_ann_fold,"labeled.json")
    # if cfg.train_dataloader.dataset.unlabeled.ann_file = osp.join(cfg.mixed_ann_fold,"unlabeled.json")

    cfg.ori_data_train_file = osp.join(cfg.data_root,cfg.data_train_ann_file)
    cfg.ori_data_val_file  = osp.join(cfg.data_root,cfg.data_val_ann_file)
    # cfg.ori_data_test_file  = osp.join(cfg.data_root,cfg.data_test_ann_file)

def patch_runner(runner,cfg):
    runner._log_dir = cfg.work_dir