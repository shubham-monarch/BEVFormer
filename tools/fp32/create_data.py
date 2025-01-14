# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from data_converter.create_gt_database import create_groundtruth_database
from data_converter import nuscenes_converter as nuscenes_converter
from data_converter import lyft_converter as lyft_converter
from data_converter import kitti_converter as kitti
from data_converter import indoor_converter as indoor
import argparse
from os import path as osp
import logging, coloredlogs
import sys
sys.path.append('.')

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)


def kitti_data_prep(root_path, info_prefix, version, out_dir):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """
    kitti.create_kitti_info_file(root_path, info_prefix)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    # logger.warning(f"=================================")    
    # logger.warning(f"[nuscenes_data_prep] called!")
    # logger.warning(f"=================================")    
    
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)
 
    return

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)
        # create_groundtruth_database(dataset_name, root_path, info_prefix,
        #                             f'{out_dir}/{info_prefix}_infos_train.pkl')


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    
    logger.error(f"=================================")    
    logger.error(f"args.dataset: {args.dataset}")
    logger.error(f"args.root_path: {args.root_path}")
    logger.error(f"args.canbus: {args.canbus}")
    logger.error(f"args.version: {args.version}")
    logger.error(f"args.max_sweeps: {args.max_sweeps}")
    logger.error(f"args.out_dir: {args.out_dir}")
    logger.error(f"args.extra_tag: {args.extra_tag}")
    logger.error(f"args.workers: {args.workers}")
    logger.error(f"=================================")    
    
    # exit(1)

    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'

        logger.info(f"=================================")    
        logger.info("args.dataset == 'nuscenes' and args.version != 'v1.0-mini'")
        logger.info(f"=================================")    
        
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        
        # exit(1)
        # test_version = f'{args.version}-test'
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     can_bus_root_path=args.canbus,
        #     info_prefix=args.extra_tag,
        #     version=test_version,
        #     dataset_name='NuScenesDataset',
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
