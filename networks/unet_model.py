import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 辅助模块：标准 UNet 卷积块 ---
class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# --- 辅助模块：下采样块 ---
class Down(nn.Module):
    """Max Pooling -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# --- 辅助模块：上采样块 (修正通道逻辑) ---
class Up(nn.Module):
    """
    Up-Convolution (ConvTranspose) -> 跳跃连接 -> DoubleConv
    in_channels 是 ConvTranspose 的输入通道 (来自前一个解码器阶段)
    skip_channels 是跳跃连接的通道 (来自编码器)
    out_channels 是本阶段最终的输出通道
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # 1. ConvTranspose2d: 输入 in_channels, 输出 in_channels // 2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 2. 融合后的总通道数: ConvT 输出 (in_channels // 2) + 跳跃连接通道 (skip_channels)
        total_conv_in = (in_channels // 2) + skip_channels
        
        # 3. 卷积块（输入 total_conv_in, 输出 out_channels）
        self.conv = DoubleConv(total_conv_in, out_channels)

    def forward(self, x1, x2):
        # x1 是瓶颈层或上一层上采样的结果 (ConvTranspose 的输入)
        # x2 是来自编码器的跳跃连接
        
        # 1. 上采样 x1
        x1 = self.up(x1)
        
        # 2. 确保尺寸兼容性 (对于 224x224 输入，通常不需要填充)
        # 我们假设输入是 2^N 尺寸 (如 224)，这里移除填充以简化代码。
        
        # 3. 沿通道维度拼接
        # x1 (ConvT Output) + x2 (Skip Connection)
        x = torch.cat([x2, x1], dim=1)
        
        # 4. 融合卷积 
        return self.conv(x)

# --- UNet 主体结构 (修正 Up 模块的初始化参数) ---
class UNet(nn.Module):
    """
    标准的 5 阶段 UNet 模型
    """
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # 初始特征图通道数 (标准 UNet)
        base_channels = 64
        
        # 1. Encoder (下采样路径)
        self.inc = DoubleConv(in_channels, base_channels)        # 1   -> 64 (x1)
        self.down1 = Down(base_channels, base_channels * 2)      # 64  -> 128 (x2)
        self.down2 = Down(base_channels * 2, base_channels * 4)  # 128 -> 256 (x3)
        self.down3 = Down(base_channels * 4, base_channels * 8)  # 256 -> 512 (x4)
        self.down4 = Down(base_channels * 8, base_channels * 8)  # 512 -> 512 (x5, Bottleneck)

        # 2. Decoder (上采样路径)
        # 修正 Up 参数: Up(ConvT_IN, SKIP_C, OUT_C)

        # Up1: x5(512) + x4(512) -> 输出 256
        self.up1 = Up(in_channels=base_channels * 8,   # ConvT 输入: 512 (来自x5)
                      skip_channels=base_channels * 8, # 跳跃连接: 512 (来自x4)
                      out_channels=base_channels * 4)  # 输出: 256
        
        # Up2: x(256) + x3(256) -> 输出 128
        self.up2 = Up(in_channels=base_channels * 4,   # ConvT 输入: 256
                      skip_channels=base_channels * 4, # 跳跃连接: 256
                      out_channels=base_channels * 2)  # 输出: 128
        
        # Up3: x(128) + x2(128) -> 输出 64
        self.up3 = Up(in_channels=base_channels * 2,   # ConvT 输入: 128
                      skip_channels=base_channels * 2, # 跳跃连接: 128
                      out_channels=base_channels)      # 输出: 64
        
        # Up4: x(64) + x1(64) -> 输出 64
        self.up4 = Up(in_channels=base_channels,       # ConvT 输入: 64
                      skip_channels=base_channels,     # 跳跃连接: 64
                      out_channels=base_channels)      # 输出: 64

        # 3. 最终输出卷积层
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1) 

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck (512 通道)

        # Decoder (跳跃连接: x4, x3, x2, x1)
        x = self.up1(x5, x4) # 512 (Bottleneck) + 512 (x4) -> 256
        x = self.up2(x, x3)  # 256 + 256 (x3) -> 128
        x = self.up3(x, x2)  # 128 + 128 (x2) -> 64
        x = self.up4(x, x1)  # 64 + 64 (x1) -> 64
        
        # 输出层 (64 -> num_classes)
        logits = self.outc(x)
        
        return logits