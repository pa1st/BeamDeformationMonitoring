"""
-*- coding: utf-8 -*-
@Time    : 2025/1/10 15:07
@Author  : PA1ST
@File    : flow_template.py.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    基于 PointNet 的 Encoder，输入点云，输出全局特征（哈希向量）。
    """

    def __init__(self, input_dim=3, feature_dim=2048):
        super(PointNetEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, feature_dim)

    def forward(self, x):
        """
        x 的形状: [batch_size, num_points, input_dim]
        """
        batch_size, num_points, _ = x.size()
        # (batch_size * num_points, input_dim)
        x = x.view(-1, x.size(-1))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 形状还原: (batch_size, num_points, 1024)，做全局汇聚
        x = x.view(batch_size, num_points, 1024)
        x = torch.max(x, dim=1)[0]  # (batch_size, 1024)

        # 全连接到最终的哈希向量 (batch_size, feature_dim)
        x = self.fc4(x)
        return x


class PointNetDecoder(nn.Module):
    """
    简单 Decoder，根据 feature_dim 输出重建点云。
    假设重建返回与输入相同形状 (num_points, input_dim)。
    """

    def __init__(self, feature_dim=2048, num_points=1024, output_dim=3):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024 * output_dim)  # 生成 num_points * output_dim 的特征

    def forward(self, x):
        """
        x 的形状: [batch_size, feature_dim]
        返回: [batch_size, num_points, output_dim]
        """
        batch_size = x.size(0)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 变形到 (batch_size, num_points, output_dim)
        x = x.view(batch_size, -1, 3)
        return x


class PointCloudAutoencoder(nn.Module):
    """
    将 Encoder + Decoder 组合成自编码器。
    """

    def __init__(self, input_dim=3, feature_dim=2048, num_points=1024, output_dim=3):
        super(PointCloudAutoencoder, self).__init__()
        self.encoder = PointNetEncoder(input_dim, feature_dim)
        self.decoder = PointNetDecoder(feature_dim, num_points, output_dim)

    def forward(self, x):
        latent_code = self.encoder(x)  # [batch_size, feature_dim]
        reconstructed = self.decoder(latent_code)  # [batch_size, num_points, output_dim]
        return latent_code, reconstructed


# ============================
# 数据加载与训练示例 (简化版本)
# ============================
def chamfer_distance_loss(pcd1, pcd2):
    """
    一个常见的点云重建损失：Chamfer Distance (简易实现示例)
    pcd1, pcd2 的形状: [batch_size, num_points, 3]
    """
    # 扩展维度 (batch_size, num_points, 1, 3) 和 (batch_size, 1, num_points, 3)
    diff_1 = pcd1.unsqueeze(2) - pcd2.unsqueeze(1)  # (batch_size, num_points, num_points, 3)
    dist_1 = torch.sum(diff_1 ** 2, dim=-1)  # (batch_size, num_points, num_points)

    # 计算每个点到对方最近点的最小距离
    dist_1_min = torch.min(dist_1, dim=2)[0]  # (batch_size, num_points)
    dist_2_min = torch.min(dist_1, dim=1)[0]  # (batch_size, num_points)

    loss = torch.mean(dist_1_min) + torch.mean(dist_2_min)
    return loss


def train_autoencoder(autoencoder, dataloader, epochs=10, lr=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.to(device)

    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0.0

        for points in dataloader:
            # points.shape: [batch_size, num_points, 3]
            points = points.to(device)

            optimizer.zero_grad()
            latent_code, reconstructed = autoencoder(points)

            # 计算 Chamfer Distance
            loss = chamfer_distance_loss(points, reconstructed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training finished!")


# ============================
# 模拟推理阶段
# ============================
if __name__ == "__main__":
    # 假设我们有一个 autoencoder 模型已经训练完成
    autoencoder = PointCloudAutoencoder()
    autoencoder.load_state_dict(torch.load("autoencoder_model.pth"))
    autoencoder.eval()

    # 示例: 对两帧点云 (T0, T1) 提取哈希向量并比较
    # 实际使用中需要载入真实的点云并进行预处理
    pcd_T0 = torch.randn(1, 1024, 3)  # batch_size=1, num_points=1024, 3D坐标
    pcd_T1 = torch.randn(1, 1024, 3)

    with torch.no_grad():
        code_T0, _ = autoencoder(pcd_T0)
        code_T1, _ = autoencoder(pcd_T1)

    # 比如可以直接计算欧几里得距离或余弦相似度
    dist = torch.norm(code_T0 - code_T1, p=2).item()
    print(f"Feature Distance between T0 & T1: {dist:.4f}")

    # 如果 dist 超过阈值，可能表示形变显著
    threshold = 10.0  # 根据经验或训练来确定
    if dist > threshold:
        print("Significant deformation suspected!")
    else:
        print("No significant deformation detected.")