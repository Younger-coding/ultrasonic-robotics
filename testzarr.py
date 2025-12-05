import torch
from torch.utils.data import DataLoader
from zarr_data import CustomSwinUnetGeometricDataset # 确保 Zarr 路径和 K 值正确

# 请替换为您的实际 Zarr 路径和 K 值
ZARR_PATH = r'D:\ultrasonic\数据集\小范围_单人.zarr'
LOOKAHEAD_K = 1 
BATCH_SIZE = 4

# 初始化 Dataset
try:
    dataset = CustomSwinUnetGeometricDataset(zarr_path=ZARR_PATH, lookahead_k=LOOKAHEAD_K)
    print(f"✅ Dataset初始化成功。有效序列对数量: {len(dataset)}")

    # 打印第一个样本的索引
    print(f"第一个有效起始索引 (t_idx): {dataset.valid_indices[0]}")
    # 打印最后一个样本的索引
    print(f"最后一个有效起始索引 (t_idx): {dataset.valid_indices[-1]}")
    
except Exception as e:
    print(f"❌ Dataset初始化失败: {e}")
    # 如果失败，请检查 Zarr 路径、数组名称 (`images`, `poses` 等) 和 `episode_ends` 的读取逻辑。
    
# 创建 DataLoader (用于后续测试)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # num_workers=0 方便调试

# 提取第一批次数据
try:
    data_batch = next(iter(loader))
    
    # 检查 I_t 和 T_t 的结构
    print("✅ 成功提取第一个批次数据。")
    print(f"I_t (image_t) 形状: {data_batch['image_t'].shape} (期望: [{BATCH_SIZE}, 1, 612, 612])")
    print(f"I_t (image_t) 类型: {data_batch['image_t'].dtype} (期望: torch.float32)")
    print(f"T_t (pose_t) 形状: {data_batch['pose_t'].shape} (期望: [{BATCH_SIZE}, 7])")
    print(f"T_tk (pose_tk) 形状: {data_batch['pose_tk'].shape} (期望: [{BATCH_SIZE}, 7])")
    
    # 检查 I_t 和 I_{t+k} 是否是不同的数据
    print(f"I_t 的均值: {data_batch['image_t'].mean().item():.4f}")
    print(f"I_tk (k={LOOKAHEAD_K}) 的均值: {data_batch['image_tk'].mean().item():.4f}")
    
except Exception as e:
    print(f"❌ DataLoader/批次提取失败: {e}")


from geometric_utils import calculate_relative_transform, quat_pos_to_matrix, project_feature_map
import torch

# 假设数据在 CPU 上
pose_t = data_batch['pose_t']
pose_tk = data_batch['pose_tk']

# ----------------------------------------------------
# 2.1 位姿转换测试
# ----------------------------------------------------

try:
    # 1. 测试单帧转换
    M_t = quat_pos_to_matrix(pose_t)
    print(f"\n✅ 2.1.1 T_t 到 4x4 矩阵转换成功。形状: {M_t.shape}")
    
    # 2. 检查 M_rel (相对变换矩阵)
    M_rel = calculate_relative_transform(pose_t, pose_tk)
    print(f"✅ 2.1.2 相对变换矩阵 M_rel 计算成功。形状: {M_rel.shape}")
    
    # 3. 几何合理性检查 (简单)
    print(f"M_rel (第一帧) 对角线元素 (应接近 1): {M_rel[0].diag()}") 

    # ----------------------------------------------------
    # 2.2 特征投影测试
    # ----------------------------------------------------
    
    # 模拟 Swin-Unet 的特征图输出 (例如 128 维度特征, 缩小到 153x153)
    BATCH_SIZE, C, H, W = pose_t.shape[0], 128, 153, 153 
    S_t_mock = torch.randn(BATCH_SIZE, C, H, W)
    
    S_prime_tk = project_feature_map(S_t_mock, M_rel)
    print(f"✅ 2.2 特征图投影成功。输出形状: {S_prime_tk.shape}")
    
except Exception as e:
    print(f"\n❌ 阶段二 几何计算验证失败: {e}")