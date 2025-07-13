import torch
import torch.nn as nn

class KinematicLoss(nn.Module):
    """结合关节位置误差与旋转平滑性的损失函数"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
            self,
            pred_joints: torch.Tensor,  # 预测的3D关节 [B, J, 3]
            target_joints: torch.Tensor,  # Kinect真值 [B, J, 3]
            pred_rotations: torch.Tensor,  # 预测的四元数 [B, J, 4]
    ) -> torch.Tensor:
        # 关节位置误差（L2损失）
        position_loss = torch.mean((pred_joints - target_joints) ** 2)

        # 旋转平滑性约束（相邻帧差分）
        rotation_diff = pred_rotations[1:] - pred_rotations[:-1]
        smoothness_loss = torch.mean(rotation_diff ** 2)

        return position_loss + self.alpha * smoothness_loss