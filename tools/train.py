import argparse
import logging
import os
import os.path as osp
import time

import mmengine
from mmengine.fileio import dump
from mmengine.registry import DefaultScope
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import get_dist_info

from mmengine import Registry
from mmengine_template import mining
# from mmengine_template.mining.utils import patch_cfg,patch_runner
from mmengine_template.mining.registry import MINERS

from mmdet import datasets
from mmdet.registry import DATASETS

from mmengine_template.models import *
from mmengine_template.registry import MODELS

from mmengine_template.datasets import datasets
from mmengine_template.registry import DATASETS

from mmdet.datasets import transforms
from mmengine_template.registry import TRANSFORMS

from mmengine_template import engine
from mmengine_template.registry import HOOKS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--al_cycles', type=int, default=3,help="The number of Active Learning cycles to perform querying data points and updating model parameters")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def build_cfg(args):

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    rank,world_size = get_dist_info()
    if rank == 0 and (os.environ.get('TIMESTAMP',None) is None):
        os.environ['TIMESTAMP'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if cfg.get("base_work_dir",None) is None:
        base_work_dir = f"{cfg.work_dir}_{os.environ.get('TIMESTAMP')}"
        cfg.base_work_dir = base_work_dir

    cfg.al_cycles = args.al_cycles

    return cfg

def main():
    # DefaultScope.get_instance(
    #     name='mmengine_template', scope_name='mmengine_template')
    args = parse_args()

    for al_cycle in range(args.al_cycles):
        cfg = build_cfg(args)
        cfg.al_cycle = al_cycle

        if cfg.experiment_name in ["ALCV"]:
            mining.utils.patch_cfg(cfg)

        os.makedirs(cfg.work_dir,exist_ok=True)

        cfg.resume = args.resume

        # mixed_dataset = DATASETS.build(cfg.unlabeled_dataset)
        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # build json file writer for unlabeled-labeled data partitions
        # write JSON file for labeled and unlabeled data...

        if cfg.experiment_name in ["ALCV"]:
            mining.utils.patch_runner(runner,cfg)

            if al_cycle == 0:
                json_writer = MINERS.build(cfg.json_writer)

            json_writer.al_cycle = al_cycle
            json_writer.set_logger(runner.logger)
            if al_cycle==0:
                json_writer.create_initial_data_partition(save_dir=cfg.work_dir)
            elif al_cycle > 0:
                scores_file = osp.join(cfg.base_work_dir,f'step_{al_cycle-1}','scores.json')
                json_writer.create_data_partitions(scores_file=scores_file,save_dir=cfg.work_dir)

            # build the unlabeled dataloader
            unlabeled_dataloader = Runner.build_dataloader(cfg.active_learning_dataloader)


        # start training
        runner.train()

        if cfg.experiment_name in ["ALCV"]:
            # write the scores for the unlabeled data into a scores.json file
            results = mining.miners.al_scores_single_gpu(runner.model,
                                 unlabeled_dataloader,
                                runner.logger,
                                 cfg.work_dir,
                                 active_cycle=-1)
            runner.logger.info(f"Saving scores.json to {cfg.work_dir}")
            dump(results, osp.join(cfg.work_dir, "scores.json"))



if __name__ == '__main__':
    main()
