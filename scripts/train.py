import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from models import KinematicNet, KinematicLoss
from utils.DataProcessor import DataProcessor


class RFIDSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, rfid_path, kinect_path):
        # 加载RFID数据
        self.rfid_df = pd.read_csv(rfid_path)

        # 移除时间戳列 (第一列)
        self.rfid_df = self.rfid_df.iloc[:, 1:]

        # 转换为数值型并处理NaN
        self.rfid_df = self.rfid_df.apply(pd.to_numeric, errors='coerce')
        self.rfid_df = self.rfid_df.dropna()

        # 数据标准化 (关键步骤!)
        self.rfid_mean = self.rfid_df.mean()
        self.rfid_std = self.rfid_df.std()
        self.rfid_df = (self.rfid_df - self.rfid_mean) / (self.rfid_std + 1e-8)

        # 加载Kinect数据
        self.kinect_data = pd.read_csv(kinect_path).values.astype(np.float32)

        # 确保数据长度匹配
        min_length = min(len(self.rfid_df), len(self.kinect_data))
        self.rfid_df = self.rfid_df.iloc[:min_length]
        self.kinect_data = self.kinect_data[:min_length]

        print(f"RFID数据形状: {self.rfid_df.shape}")
        print(f"Kinect数据形状: {self.kinect_data.shape}")

    def __getitem__(self, idx):
        rfid_sample = self.rfid_df.iloc[idx].values.astype(np.float32)
        kinect_sample = self.kinect_data[idx]
        return rfid_sample, kinect_sample

    def __len__(self):
        # 长度现在由 self.rfid_df 的长度决定
        return len(self.rfid_df)


def train():
    # 加载配置
    with open("../configs/model.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    # 加载骨架初始结构和父节点结构
    initial_skeleton = torch.tensor(np.load(cfg["initial_skeleton"]), dtype=torch.float32).cuda()
    parent_indices = np.load("../data/skeleton_parent.npy").tolist()

    model = KinematicNet(
        input_shape=36,
        num_joints=12,
        hidden_size=256,
        initial_offsets=initial_skeleton,
        parent_indices=parent_indices,
    ).cuda()

    # 暂时禁用旋转平滑约束
    criterion = KinematicLoss(alpha=0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率

    # 数据加载 - 使用修复后的数据集类
    dataset = RFIDSkeletonDataset("../data/RFID_628/dataset/train/2_walking/all.csv",
                                  "../sim_data/simulated_data/walking/kinect_3d.csv")

    # 添加数据检查点
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("数据集为空！请检查数据路径")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (rfid, kinect) in enumerate(loader):
            # 确保数据在GPU上
            rfid = rfid.cuda()
            kinect = kinect.cuda()

            # 模型前向传播
            pred_joints, pred_rot = model(rfid)

            # 计算损失
            loss = criterion(pred_joints, kinect.view(-1, 12, 3), pred_rot)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 打印数值稳定的损失
            # if not torch.isnan(loss):
            #     print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            epoch_loss += loss.item()
            num_batches += 1

            # 每10个batch打印一次进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch} complete, Average Loss: {avg_loss:.4f}")

    print("训练完成！")

if __name__ == "__main__":
    train()