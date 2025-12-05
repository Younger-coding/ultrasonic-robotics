import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import os
from typing import List
import torch.nn.functional as F 

# 假设 Lookahead K=1 (即使用相邻帧 I_t 和 I_{t+1})
LOOKAHEAD_K = 1

# 目标是 224x224，用于 Resize
target_size = 224

class CustomSwinUnetGeometricDataset(Dataset):
    """
    定制化的 Zarr 数据集，用于多视角几何一致性训练。
    同时加载 I_t, T_t, I_{t+k}, T_{t+k}。
    """
    def __init__(self, zarr_path: str, lookahead_k: int = LOOKAHEAD_K, transform=None):
        
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr 路径未找到: {zarr_path}")

        # 使用 'r' 模式读取 Zarr
        self.root = zarr.open(zarr_path, mode='r')
        self.data_group = self.root['data']
        self.meta_group = self.root['meta']

        # 1. 引用核心数组
        self.images = self.data_group['img_current']
        self.poses = self.data_group['current_pose'] 
        self.episode_ends = self.meta_group['episode_ends'][:] # 强制加载到内存

        self.transform = transform
        self.lookahead_k = lookahead_k
        
        # 2. 预处理：生成有效的起始索引列表 (t)
        self.valid_indices = self._preprocess_indices()
        
        print(f"原始样本总数: {self.images.shape[0]}")
        print(f"有效序列 (t, t+k) 数量: {len(self.valid_indices)}")
        print(f"使用的 Lookahead (k): {self.lookahead_k}")

    def _preprocess_indices(self) -> List[int]:
        """确定所有有效的 t 索引，确保 t+k 位于同一剧集内。"""
        valid_indices = []
        episode_start = 0
        
        for end_idx in self.episode_ends:
            max_t_idx = end_idx - self.lookahead_k
            
            for t_idx in range(episode_start, max_t_idx):
                valid_indices.append(t_idx)
            
            episode_start = end_idx

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, sequence_idx: int):
        # 1. 获取实际的 t 和 t+k 数组索引
        t_idx = self.valid_indices[sequence_idx]
        tk_idx = t_idx + self.lookahead_k
        
        # 2. 提取数据 (NumPy arrays)
        # 形状: (H, W) uint8
        np_image_t = self.images[t_idx] 
        np_image_tk = self.images[tk_idx]
        
        # 形状: (7,) float32
        np_pose_t = self.poses[t_idx]
        np_pose_tk = self.poses[tk_idx]

        # 3. 归一化和转换为 PyTorch Tensor (CxHxW)
        # (H, W) -> (1, H, W). 归一化到 [0, 1]
        image_t = torch.from_numpy(np_image_t).float().unsqueeze(0) / 255.0
        image_tk = torch.from_numpy(np_image_tk).float().unsqueeze(0) / 255.0
        
        # 4. 位姿转换为 PyTorch Tensor
        pose_t = torch.from_numpy(np_pose_t).float()
        pose_tk = torch.from_numpy(np_pose_tk).float()

        # 5. Resize 逻辑 (在 CPU 上)
        # F.interpolate 要求 BxCxHxW 格式。
        if image_t.shape[1] != target_size:
            # 增加一个 Batch 维度: (1, H, W) -> (1, 1, H, W)
            image_t_bchw = image_t.unsqueeze(0) 
            image_tk_bchw = image_tk.unsqueeze(0)
            
            # 使用 F.interpolate 进行缩放
            image_t = F.interpolate(image_t_bchw, size=(target_size, target_size), 
                                    mode='bilinear', align_corners=False)
            image_tk = F.interpolate(image_tk_bchw, size=(target_size, target_size), 
                                    mode='bilinear', align_corners=False)

            # 移除 Batch 维度: (1, 1, 224, 224) -> (1, 224, 224)
            image_t = image_t.squeeze(0)
            image_tk = image_tk.squeeze(0)

        # 6. 返回最终数据
        return {
            'image_t': image_t, 
            'pose_t': pose_t, 
            'image_tk': image_tk, 
            'pose_tk': pose_tk
        }