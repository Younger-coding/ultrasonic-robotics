import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
# ⚠️ 替换导入：改为导入新的训练器函数
import trainer_geometric # 假设您已将 trainer_synapse 改名为此
from config import get_config

# 导入您的定制数据集类和 Lookahead K
# 假设 zarr_data.py 中有 CustomSwinUnetGeometricDataset 和 LOOKAHEAD_K
from zarr_data import CustomSwinUnetGeometricDataset, LOOKAHEAD_K 

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

parser = argparse.ArgumentParser()
# ⚠️ 新增参数：用于指定 Zarr 文件的完整路径
parser.add_argument('--zarr_path', type=str,
                     default=r'D:\ultrasonic\数据集\小范围_单人.zarr', 
                     help='Path to the Zarr dataset file')
# ⚠️ 移除 root_path/list_dir，或保留但不使用
parser.add_argument('--root_path', type=str,
                     default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                     default='GeometricConsistency', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                     default='./lists/lists_Synapse', help='list dir')

# ⚠️ num_classes/n_class 现在是特征维度，而不是分割类别数
parser.add_argument('--num_classes', type=int,
                     default=128, help='output channel of network (Feature Dimension for Self-Supervision)')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                     default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                     default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                     default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                     help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                     help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                     default=224, help='input patch size of network input (Use Zarr native size 612)')
parser.add_argument('--seed', type=int,
                     default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                     help='no: no cache, '
                           'full: cache all data, '
                           'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                     help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                     help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# parser.add_argument("--dataset_name", default="datasets")
parser.add_argument("--n_class", default=128, type=int) # ⚠️ 将默认特征维度改为 128
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)

args = parser.parse_args()

# Synapse 数据集相关的路径处理
# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ⚠️ 配置现在应该反映新的特征维度
    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            # 'root_path': args.root_path, # 不使用
            # 'list_dir': f'./lists/{args.dataset}', # 不使用
            'num_classes': args.n_class, # 使用新的特征维度
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
        
    # ⚠️ 更新参数：使用新的特征维度作为模型的输出通道数
    args.num_classes = dataset_config[dataset_name]['num_classes'] 
    # args.root_path = dataset_config[dataset_name]['root_path'] # 不再重要
    # args.list_dir = dataset_config[dataset_name]['list_dir'] # 不再重要

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # ⚠️ 初始化 Swin-Unet 模型，输出通道数是特征维度 (128)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    # ⚠️ 替换训练器函数
    trainer_geometric.trainer_geometric_consistency(args, net, args.output_dir)

# 示例运行命令（使用特征维度 128）
# python train.py --output_dir ./model_out/geometric --dataset GeometricConsistency --img_size 224 --batch_size 16 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --num_classes 128 --zarr_path 'D:\ultrasonic\数据集\小范围_单人.zarr' --num_workers 0