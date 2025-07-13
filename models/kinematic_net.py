# kinematic_net.py
import torch
import torch.nn as nn
from typing import Tuple, List

def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """将四元数 [B, 4] 转换为旋转矩阵 [B, 3, 3]"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.size(0)

    R = torch.zeros(B, 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R

class ForwardKinematicsLayer(nn.Module):
    """前向运动学（FK）层，将旋转角转换为3D关节坐标"""
    def __init__(self, initial_skeleton: torch.Tensor, parent_indices: List[int]):
        super().__init__()
        self.register_buffer("initial_skeleton", initial_skeleton)  # shape [J, 3]
        self.parent_indices = parent_indices

    def forward(self, rotations: torch.Tensor) -> torch.Tensor:
        """
        rotations: [B, J, 4] 四元数
        returns: [B, J, 3] 每个关节的3D位置
        """
        B, J, _ = rotations.shape
        joints = torch.zeros(B, J, 3, device=rotations.device)

        for j in range(J):
            parent = self.parent_indices[j]
            q_j = rotations[:, j, :]  # [B, 4]
            R = quat_to_matrix(q_j)   # [B, 3, 3]
            if parent == -1:
                joints[:, j, :] = 0.0
            else:
                offset = self.initial_skeleton[j] - self.initial_skeleton[parent]  # [3]
                offset = offset.expand(B, 3).unsqueeze(-1)  # [B, 3, 1]
                rotated = torch.bmm(R, offset).squeeze(-1)  # [B, 3]
                joints[:, j, :] = joints[:, parent, :] + rotated
        return joints


class KinematicNet(nn.Module):
    def __init__(
            self,
            input_shape: int = 36,  # 修正为实际输入维度
            num_joints: int = 12,
            hidden_size: int = 256,
            initial_offsets: torch.Tensor = None,
            parent_indices: List[int] = None,
    ):
        super().__init__()
        self.num_joints = num_joints

        # 使用全连接网络替代GRU
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_joints * 4),
        )

        self.fk_layer = ForwardKinematicsLayer(initial_offsets, parent_indices)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        rot = self.decoder(x).view(-1, self.num_joints, 4)
        rot = rot / (torch.norm(rot, dim=-1, keepdim=True) + 1e-8)  # 防止除零
        joints_3d = self.fk_layer(rot)
        return joints_3d, rot
