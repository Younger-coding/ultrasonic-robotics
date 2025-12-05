import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# ====================================================================
# 导入模型和工具
# ====================================================================

# Swin-Unet 相关的导入 
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

# UNet 相关的导入 (确保 unet_model.py 文件存在)
from networks.unet_model import UNet 

# 训练器和数据集导入 (根据您的实际文件名)
import trainer_geometric
from zarr_data import CustomSwinUnetGeometricDataset # 导入 Zarr 数据集类

import multiprocessing
# 强制使用 spawn 启动方法，解决 PyTorch DataLoader 在 Windows 上的兼容性问题
try:
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ====================================================================
# 参数定义
# ====================================================================

parser = argparse.ArgumentParser()
# --- Zarr 数据集路径 ---
parser.add_argument('--zarr_path', type=str,
                     default=r'D:\ultrasonic\数据集\小范围_单人.zarr', 
                     help='Path to the Zarr dataset file')
# --- 模型选择参数 ---
parser.add_argument(
    '--model_type', type=str, default='swinunet', 
    choices=['swinunet', 'unet'], 
    help='选择要使用的模型类型: swinunet (默认) 或 unet'
) 
# --- 其他通用参数  ---
parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str, default='GeometricConsistency', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=128, help='output channel of network (Feature Dimension for Self-Supervision)')
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
parser.add_argument("--num_workers", default=0, type=int) # 保持 0，解决内存溢出问题
parser.add_argument("--eval_interval", default=1, type=int)


# ====================================================================
# 模型初始化函数 
# ====================================================================
def initialize_model(args, config=None):
    """根据 args.model_type 初始化模型"""
    if args.model_type.lower() == 'unet':
        # 传统 UNet 初始化
        net = UNet(in_channels=1, 
                   num_classes=args.num_classes
                  ).cuda()
        
        print(f"Loaded Traditional UNet. Output Features: {args.num_classes}")
        return net
    
    elif args.model_type.lower() == 'swinunet':
        # Swin-Unet 初始化
        if not config:
             # 如果没有提供 config，尝试再次加载（防止外部调用时遗漏）
             config = get_config(args) 

        # 初始化 Swin-Unet
        net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

        # 加载 Swin-Unet 预训练权重
        try:
            print("---start load pretrained modle of swin encoder---")
            net.load_from(config) 
            print("---load pretrained modle successful---")
        except Exception as e:
            print(f"Swin-Unet 预训练权重加载失败或已跳过: {e}. 将从头开始训练。")
        
        print(f"Loaded Swin-Unet. Output Features: {args.num_classes}")
        return net
    
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

# ====================================================================
# 主程序入口
# ====================================================================
if __name__ == "__main__":
    args = parser.parse_args()

    # ----------------------------------------------------
    # 确定性设置
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # 模型和配置加载
    # ----------------------------------------------------
    config = None
    if args.model_type.lower() == 'swinunet':
        # 仅在 Swin-Unet 模式下加载配置
        config = get_config(args)
    
    # ⚠️ 关键修改：调用统一的模型初始化函数
    net = initialize_model(args, config)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ----------------------------------------------------
    # 数据加载 (Zarr)
    # ----------------------------------------------------
    # 注意：Zarr 数据加载由 CustomSwinUnetGeometricDataset 处理
    db_train = CustomSwinUnetGeometricDataset(zarr_path=args.zarr_path, 
                                              lookahead_k=1, # 假设 Lookahead K=1
                                              transform=None)
    
    print(f"The length of train set (valid pairs) is: {len(db_train)}")
    
    # 计算最大迭代次数
    num_train_pairs = len(db_train)
    if args.batch_size > 0:
        num_iterations_per_epoch = num_train_pairs // args.batch_size
        args.max_iterations = args.max_epochs * num_iterations_per_epoch
        print(f"{num_iterations_per_epoch} iterations per epoch. {args.max_iterations} max iterations.")
    else:
        print("Batch size is zero or not positive, training loop will exit immediately.")
        exit()

    # ⚠️ DataLoader：必须保持 num_workers=0 (低内存环境)
    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)

    # ----------------------------------------------------
    # 训练入口
    # ----------------------------------------------------
    
    # 调用训练器函数
    trainer_geometric.trainer_geometric_consistency(args, net, args.output_dir, train_loader)

# 示例运行命令（使用特征维度 8）
# python train_new.py --output_dir ./model_out/geometric --dataset GeometricConsistency --img_size 224 --batch_size 16 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --num_classes 8 --zarr_path 'D:\ultrasonic\数据集\小范围_单人.zarr' --num_workers 0