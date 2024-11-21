# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmdet.utils.logger import get_logger_
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets import custom_build_dataset
def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)
    logger_ = get_logger_(cfg.log_level)

    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #assert len(dataset)==1s
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        ) for ds in dataset
    ]

    # keys = {img_metas,gt_boxes_3d,img,semantic_indices}
    logger_.info("================================================")
    
    first_dataloader = data_loaders[0]
    logger_.info("Iterating over first_dataloader.dataset:")
    logger_.info(f"type(first_dataloader.dataset): {type(first_dataloader.dataset)}")
    
    for i, data in enumerate(first_dataloader.dataset):
        logger_.warning(f"# Sample {i}:")
        for k, v in data.items():
            if k == 'img_metas':
                logger_.info(f"type(img_metas): {type(v)}")
                logger_.info(f"len(img_metas): {len(v)}")
                # if i == 0:
                    # logger_.info(f"img_metas[0]: {v[0]}")
        # for key, value in data.items():
        #     logger_.info(f"key: {key} type: {type(key)}")
        #     logger_.info(f"type(value): {type(value)}")
            
            
        #     if issinstance(value, (list, tuple, dict)):
        #         logger_.info(f"  {key}: {type(value)}")
        #     else:
        #         logger_.info(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else None}")
        if i >= 10:  # Only show first 3 samples
            break

    


    # first_batch = next(iter(first_dataloader))
    
    # logger_.info(f"dir(first_batch): {dir(first_batch)}")
    # logger_.info("First batch keys: %s", first_batch.keys())

    # img_metas = first_batch['img_metas']
    # logger_.info(f"type(img_metas): {type(img_metas)}")
    # logger_.info(f"len(img_metas): {len(img_metas)}")
    # # logger_.info(f"img_metas[0]: {img_metas[0]}")

    # for k, v in first_batch.items():
    #     if isinstance(v, torch.Tensor):
    #         logger_.warning("Key: %s", k)
    #         logger_.warning("Shape: %s", v.shape)
    #         logger_.warning("Dtype: %s", v.dtype) 
    #         logger_.warning("Value: %s", v)
    #         logger_.warning("------------------------")
    #     else:
    #         logger_.warning("Key: %s", k)
    #         logger_.warning("Type: %s", type(v))
    #         logger_.warning("Value: %s", v)
    #         logger_.warning("------------------------")

    # Method 2: Dataset info
    # logger_.warning("\nDataset info:")
    # logger_.warning("Number of batches: %d", len(first_dataloader))
    # logger_.warning("Batch size: %d", cfg.data.samples_per_gpu)
    # logger_.warning("Dataset size: %d", len(ds))
    # logger_.warning(f"Number of data loaders: {len(data_loaders)}")
    # logger_.warning(f"Type of data_loaders: {type(data_loaders)}")
    
    # if len(data_loaders) > 0:
    #     first_loader = data_loaders[0]
    #     logger_.warning(f"dir")
    #     logger_.warning("\nFirst data loader details:")
    #     logger_.warning(f"Type: {type(first_loader)}")
    #     logger_.warning(f"Batch size: {first_loader.batch_size}")
    #     logger_.warning(f"Number of workers: {first_loader.num_workers}")
    #     logger_.warning(f"Sampler type: {type(first_loader.sampler)}")
    #     logger_.warning(f"Dataset type: {type(first_loader.dataset)}")
    #     logger_.warning(f"Length of dataset: {len(first_loader.dataset)}")
    logger_.info("================================================\n")

    logger_.info("================================================")
    logger_.info(f"distributed: {distributed}")
    logger_.info("================================================\n")

    # exit(1)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    # print("================================================")
    # print("[mmdet_train.py -> validation started]")
    # print("================================================\n")

    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # print("================================================")
    # print("[mmdet_train.py -> validation finished]")
    # print("================================================\n")

    # exit(1)

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    print("================================================")
    print(f"cfg.workflow: {cfg.workflow}")
    print(f"type(data_loaders): {type(data_loaders)}")
    print(f"data_loaders: {data_loaders}")
    print("================================================\n")

    # exit(1)
    runner.run(data_loaders, cfg.workflow)

    # exit(1)    