import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np

# ⚠️ 注意: 这里的实现是复杂且高度依赖您的数据的。

# 假设您的相机内参 (Intrinsics) 是全局已知的 Tensor 或参数
# 示例内参 (需要您用实际值替换)
# [fx, 0, cx]
# [0, fy, cy]
# [0, 0, 1]
CAMERA_INTRINSICS = torch.tensor([
    [500.0, 0.0, 306.0],
    [0.0, 500.0, 306.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)


def quat_pos_to_matrix(pose_7d: torch.Tensor) -> torch.Tensor:
    """
    将 7D 位姿 (x, y, z, qx, qy, qz, qw) 转换为 4x4 齐次变换矩阵。
    
    Args:
        pose_7d: [B, 7] 的 Tensor，包含 (position + quaternion)。
        
    Returns:
        [B, 4, 4] 的齐次变换矩阵 M_t。
    """
    B = pose_7d.shape[0]
    device = pose_7d.device
    
    # TODO: 1. 将四元数 (qx, qy, qz, qw) 转换为 3x3 旋转矩阵 R
    # ⚠️ 必须使用 NumPy/SciPy 的四元数转换函数或 PyTorch Geometric/PyTorch3D 等几何库。
    
    # 占位符: 假设旋转矩阵 R_mat = [B, 3, 3]
    R_mat = torch.eye(3, device=device).repeat(B, 1, 1) 
    
    # 2. 提取平移向量 t = (x, y, z)
    t = pose_7d[:, :3].unsqueeze(-1) # [B, 3, 1]
    
    # 3. 构建 4x4 齐次矩阵 M
    M = torch.zeros((B, 4, 4), device=device)
    M[:, :3, :3] = R_mat
    M[:, :3, 3] = t.squeeze(-1)
    M[:, 3, 3] = 1.0
    return M


def calculate_relative_transform(pose_t: torch.Tensor, pose_tk: torch.Tensor) -> torch.Tensor:
    """
    计算 T_t 到 T_{t+k} 的相对变换矩阵 M_rel = M_{t+k} @ inv(M_t)。
    """
    # 1. 转换为 4x4 矩阵
    M_t = quat_pos_to_matrix(pose_t)
    M_tk = quat_pos_to_matrix(pose_tk)
    
    # 2. 计算逆矩阵 M_t_inv (旋转矩阵的逆是转置)
    M_t_inv = torch.inverse(M_t) # 或对于欧式变换使用更快的 M_t_inv = M_t.transpose(-1, -2) 
    
    # 3. 相对变换
    M_rel = torch.bmm(M_tk, M_t_inv)
    return M_rel


def project_feature_map(S_t: torch.Tensor, M_rel: torch.Tensor) -> torch.Tensor:
    """
    将特征图 S_t 通过相对变换 M_rel (和相机内参) 投影到 I_{t+k} 的视图。
    
    Args:
        S_t: [B, C, H, W] 的特征图。
        M_rel: [B, 4, 4] 的相对变换矩阵。
        
    Returns:
        [B, C, H, W] 的投影特征图 S'_tk。
    """
    B, C, H, W = S_t.shape
    device = S_t.device
    
    # ⚠️ TODO: 核心几何投影逻辑 (Grid Sampler)
    # 这部分涉及将 M_rel, 相机内参, 图像坐标系进行结合，生成一个用于 grid_sample 的采样网格。
    
    # 占位符: 如果没有实现投影，返回 S_t 的克隆
    if M_rel.shape[1] != 4:
         print("Warning: Geometric projection not fully implemented. Returning clone.")
         return S_t.clone()

    # 假设我们已经通过几何计算得到了一个 [B, H, W, 2] 的归一化采样网格 (Grid)
    # 这里的 grid 生成是难度最大的部分，需要根据实际相机模型和位姿计算
    # grid = generate_sampling_grid(M_rel, H, W, K=CAMERA_INTRINSICS.to(device))
    
    # 占位符 Grid (返回一个无变换的 Grid)
    grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1).to(device), [B, C, H, W], align_corners=True)
    
    # F.grid_sample(input, grid) 进行采样
    S_prime_tk = F.grid_sample(S_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return S_prime_tk