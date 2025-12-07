import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import get_config
# Swin-Unet with contrastive/IIC heads
from networks.vision_transformer import SwinUnetWithHeads as ViT_seg
# Optional baseline UNet
from networks.unet_model import UNet

import trainer_geometric
from zarr_data import CustomSwinUnetGeometricDataset
import zarr_data  # to adjust module-level aug/flow switches

import multiprocessing

# Force spawn to avoid PyTorch DataLoader issues on Windows
try:
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ---------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--zarr_path', type=str,
                    default=r'D:\ultrasonic\数据集\小范围_单人.zarr',
                    help='Path to the Zarr dataset file')
# Model choice
parser.add_argument('--model_type', type=str, default='swinunet',
                    choices=['swinunet', 'unet'],
                    help='Model type to use: swinunet (default) or unet')
# Common training args
parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str, default='GeometricConsistency', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=128, help='output channel of network (legacy, used by UNet)')
parser.add_argument('--output_dir', type=str, default='./model_out/default', help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='cache mode')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument("--n_class", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)  # keep 0 to avoid OOM on Windows
parser.add_argument("--eval_interval", default=1, type=int)
# Self-supervised / contrastive / IIC args
parser.add_argument("--lambda_contrast", type=float, default=1.0, help="weight for contrastive loss")
parser.add_argument("--lambda_iic", type=float, default=0.5, help="weight for IIC loss")
parser.add_argument("--feat_dim", type=int, default=128, help="projection head output dim for contrastive")
parser.add_argument("--num_clusters", type=int, default=16, help="cluster head output dim (IIC categories)")
parser.add_argument("--lookahead_k", type=int, default=1, help="t+k frame distance")
parser.add_argument("--enable_flow", action='store_true', help="use optical flow to warp features")
parser.add_argument("--max_rotate", type=float, default=360.0, help="max rotation degree for strong aug")
parser.add_argument("--noise_std", type=float, default=0.1, help="gaussian noise std for strong aug")


# ---------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------
def initialize_model(args, config=None):
    """Initialize model according to args.model_type."""
    if args.model_type.lower() == 'unet':
        net = UNet(in_channels=1, num_classes=args.num_classes).cuda()
        print(f"Loaded Traditional UNet. Output Features: {args.num_classes}")
        return net

    if args.model_type.lower() == 'swinunet':
        if not config:
            config = get_config(args)

        net = ViT_seg(
            config,
            img_size=args.img_size,
            feat_dim=args.feat_dim,
            num_clusters=args.num_clusters
        ).cuda()

        try:
            print("---start load pretrained model of swin encoder---")
            net.backbone.load_from(config)
            print("---load pretrained model successful---")
        except Exception as e:
            print(f"Swin-Unet pretrained weight load failed/skip: {e}. Training from scratch.")

        print(f"Loaded Swin-UnetWithHeads. proj dim: {args.feat_dim}, clusters: {args.num_clusters}")
        return net

    raise ValueError(f"Unsupported model type: {args.model_type}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()

    # Determinism
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    # Load config if using swinunet
    config = None
    if args.model_type.lower() == 'swinunet':
        config = get_config(args)

    # Initialize model
    net = initialize_model(args, config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set augmentation/flow knobs in dataset module
    zarr_data.MAX_ROTATE_DEG = args.max_rotate
    zarr_data.NOISE_STD = args.noise_std
    zarr_data.ENABLE_FLOW = args.enable_flow

    # Dataset & loader
    db_train = CustomSwinUnetGeometricDataset(
        zarr_path=args.zarr_path,
        lookahead_k=args.lookahead_k,
        transform=None
    )
    print(f"The length of train set (valid pairs) is: {len(db_train)}")

    num_train_pairs = len(db_train)
    if args.batch_size > 0:
        num_iterations_per_epoch = num_train_pairs // args.batch_size
        args.max_iterations = args.max_epochs * num_iterations_per_epoch
        print(f"{num_iterations_per_epoch} iterations per epoch. {args.max_iterations} max iterations.")
    else:
        print("Batch size is zero or not positive, training loop will exit immediately.")
        exit()

    train_loader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Training entry
    trainer_geometric.trainer_geometric_consistency(args, net, args.output_dir, train_loader)

# 示例运行命令（特征维度128，聚类16）:
# python train_new.py --output_dir ./model_out/geometric --dataset GeometricConsistency --img_size 224 --batch_size 2 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --num_clusters 16 --feat_dim 128 --zarr_path "D:\ultrasonic\数据集\小范围_单人.zarr" --num_workers 0 --enable_flow --lambda_contrast 1.0 --lambda_iic 0.5 --max_epochs 1
