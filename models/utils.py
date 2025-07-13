import torch

def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """PyTorch 2.x 兼容的四元数转旋转矩阵"""
    quat = quat.to(dtype=torch.float32)  # 强制类型
    return torch.nn.functional.normalize(quat, p=2, dim=-1)  # 单位化处理