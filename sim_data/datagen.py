import numpy as np
import pandas as pd
import yaml
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json
import random


class RFIDKinectDataGenerator:
    def __init__(self, config_path="data.yml", output_dir="./simulated_data"):
        # 加载配置文件
        self.load_config(config_path)

        # 基础参数设置
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 从配置文件中获取参数
        self.num_antennas = self.config['shape'][0]
        self.time_steps = self.config['shape'][1]
        self.num_tags = self.config['shape'][2]

        # 标签ID映射
        self.tag_ids = [self.config['tags'][i] for i in sorted(self.config['tags'].keys())]
        self.tag_names = {v: k for k, v in self.config['tags'].items()}

        # 动作映射
        self.action_map = {int(k): v for k, v in self.config['names'].items()}

        # 物理参数
        self.c = 3e8  # 光速
        self.freq_min = 902e6  # 最小频率 (902MHz)
        self.freq_max = 928e6  # 最大频率 (928MHz)
        self.num_channels = 50  # 信道数量

        # 设置标签和天线位置
        self.setup_positions()

        # 预定义关节父子关系 (根据论文图10)
        self.joint_hierarchy = {
            0: [],  # Pelvis (根节点)
            1: [0],  # Left hip
            2: [1],  # Left knee
            3: [0],  # Right hip
            4: [3],  # Right knee
            5: [0],  # Left shoulder
            6: [5],  # Left elbow
            7: [6],  # Left wrist
            8: [0],  # Right shoulder
            9: [8],  # Right elbow
            10: [9],  # Right wrist
            11: [0]  # Neck
        }

        # 动作参数库
        self.action_params = {
            "still": {"amplitude": 0.0, "frequency": 0.0},
            "squatting": {"amplitude": 0.4, "frequency": 0.3},
            "walking": {"amplitude": 0.3, "frequency": 1.0},
            "twisting": {"amplitude": 0.5, "frequency": 0.8},
            "drinking": {"amplitude": 0.2, "frequency": 0.5},
            "boxing": {"amplitude": 0.6, "frequency": 1.2}
        }

        # 计算持续时间 (基于时间步数)
        self.duration = self.time_steps / 8  # 8Hz采样率 (32步对应4秒)

    def load_config(self, config_path):
        """加载YAML配置文件"""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")

    def setup_positions(self):
        """设置标签和天线位置 (根据论文图12和您的配置文件)"""
        # 人体关节标签位置 (初始T-pose)
        self.tag_positions = np.array([
            [0, 0, 0.95],  # 0: Pelvis (腰部)
            [-0.2, 0, 0.9],  # 1: Left hip
            [-0.2, 0, 0.6],  # 2: Left knee
            [0.2, 0, 0.9],  # 3: Right hip
            [0.2, 0, 0.6],  # 4: Right knee
            [-0.3, 0, 1.1],  # 5: Left shoulder (中1)
            [-0.5, 0, 1.1],  # 6: Left elbow (左1)
            [-0.7, 0, 1.1],  # 7: Left wrist (左2)
            [0.3, 0, 1.1],  # 8: Right shoulder (中2)
            [0.5, 0, 1.1],  # 9: Right elbow (右1)
            [0.7, 0, 1.1],  # 10: Right wrist (右2)
            [0, 0, 1.2]  # 11: Neck (中3)
        ])

        # 天线位置 (根据配置文件的1个天线)
        self.antenna_positions = np.array([
            [0, 2.0, 1.2],  # 前方天线
        ])

    def generate_motion(self, action_id):
        """生成运动轨迹 (基于配置文件中的动作类型)"""
        action_name = self.action_map[action_id]
        params = self.action_params[action_name]

        # 生成时间序列 (Kinect采样率)
        t_kinect = np.linspace(0, self.duration, self.time_steps)

        # 初始化运动数据
        motion_data = np.zeros((len(t_kinect), self.num_tags, 3))

        # 基础位置
        motion_data += self.tag_positions[np.newaxis, :, :]

        # 应用动作
        if action_name == "still":
            pass  # 静止状态，无额外运动

        elif action_name == "squatting":
            # 下蹲运动
            squat_motion = params["amplitude"] * (1 + np.sin(2 * np.pi * params["frequency"] * t_kinect - np.pi / 2))
            motion_data[:, :, 2] -= squat_motion[:, np.newaxis]

        elif action_name == "walking":
            # 行走运动
            motion_data[:, :, 0] += params["amplitude"] * np.sin(
                2 * np.pi * params["frequency"] * t_kinect
            )[:, np.newaxis]

        elif action_name == "twisting":
            # 扭动运动
            twist = params["amplitude"] * np.sin(2 * np.pi * params["frequency"] * t_kinect)
            for i in range(len(t_kinect)):
                rotation = np.array([
                    [np.cos(twist[i]), 0, np.sin(twist[i])],
                    [0, 1, 0],
                    [-np.sin(twist[i]), 0, np.cos(twist[i])]
                ])
                motion_data[i] = np.dot(motion_data[i], rotation)

        elif action_name == "drinking":
            # 喝水动作 (手臂运动)
            arm_motion = params["amplitude"] * (1 - np.cos(2 * np.pi * params["frequency"] * t_kinect))
            motion_data[:, 6, 0] -= arm_motion  # 左肘
            motion_data[:, 6, 2] += arm_motion
            motion_data[:, 9, 0] += arm_motion  # 右肘
            motion_data[:, 9, 2] += arm_motion

        elif action_name == "boxing":
            # 拳击动作
            punch_left = params["amplitude"] * np.sin(2 * np.pi * params["frequency"] * t_kinect)
            punch_right = params["amplitude"] * np.sin(2 * np.pi * params["frequency"] * t_kinect + np.pi)
            motion_data[:, 7, 0] -= punch_left  # 左腕
            motion_data[:, 10, 0] += punch_right  # 右腕

        return motion_data

    def calculate_phase(self, tag_pos, ant_pos, freq):
        """计算RFID相位 (根据论文公式1)"""
        distance = np.linalg.norm(tag_pos - ant_pos)

        # 基础相位计算
        phase = (4 * np.pi * distance * freq) / self.c

        # 添加标签和天线相位偏移 (论文中Φ_tag和Φ_a)
        phase_offset = np.random.uniform(0, 2 * np.pi)

        # 添加多径效应噪声
        multipath_noise = 0.1 * np.random.randn()

        # 最终相位 (模2π)
        return (phase + phase_offset + multipath_noise) % (2 * np.pi)

    def simulate_rfid_data(self, motion_data, action_id):
        """模拟RFID数据采集 (包含相位扭曲和缺失数据)"""
        # 生成RFID时间序列 (更高采样率)
        t_rfid = np.linspace(0, self.duration, int(20 * self.duration))  # 20Hz采样率

        # 插值运动数据到RFID时间点
        rfid_positions = np.zeros((len(t_rfid), self.num_tags, 3))
        for j in range(self.num_tags):
            for dim in range(3):
                interp_func = interp1d(
                    np.linspace(0, self.duration, len(motion_data)),
                    motion_data[:, j, dim],
                    kind='linear'
                )
                rfid_positions[:, j, dim] = interp_func(t_rfid)

        # 生成RFID数据 (天线×时间×标签)
        rfid_data = []

        for a, ant_pos in enumerate(self.antenna_positions):
            antenna_data = []
            for t_idx in range(len(t_rfid)):
                time_data = []
                for tag_idx in range(self.num_tags):
                    # 随机选择频率信道 (论文中50个信道)
                    freq = np.random.uniform(self.freq_min, self.freq_max)

                    # 计算相位
                    phase = self.calculate_phase(
                        rfid_positions[t_idx, tag_idx],
                        ant_pos,
                        freq
                    )

                    # 添加时间戳、天线ID、标签ID、相位值和频率信道
                    time_data.append({
                        "timestamp": t_rfid[t_idx],
                        "antenna_id": a,
                        "tag_id": self.tag_ids[tag_idx],
                        "phase": phase,
                        "freq_channel": int((freq - self.freq_min) / (self.freq_max - self.freq_min) * 49)
                    })

                # 随机缺失数据 (模拟ALOHA协议)
                if np.random.rand() < 0.7:  # 70%缺失率
                    # 随机选择一个标签保留，其他置为NaN
                    keep_idx = np.random.randint(0, self.num_tags)
                    for i in range(self.num_tags):
                        if i != keep_idx:
                            time_data[i]["phase"] = np.nan

                antenna_data.append(time_data)
            rfid_data.append(antenna_data)

        return rfid_data, rfid_positions

    def save_data(self, action_id, kinect_data, rfid_data):
        """保存模拟数据到文件"""
        action_name = self.action_map[action_id]
        output_dir = os.path.join(self.output_dir, action_name)
        os.makedirs(output_dir, exist_ok=True)

        # 保存Kinect数据 (3D坐标)
        kinect_df = pd.DataFrame(
            kinect_data.reshape(kinect_data.shape[0], -1),
            columns=[f"joint{j}_{coord}" for j in range(self.num_tags) for coord in ["x", "y", "z"]]
        )
        kinect_path = os.path.join(output_dir, "kinect_3d.csv")
        kinect_df.to_csv(kinect_path, index=False)

        # 保存RFID数据 (原始格式)
        rfid_records = []
        for ant_data in rfid_data:
            for time_data in ant_data:
                for tag_data in time_data:
                    if not np.isnan(tag_data["phase"]):
                        rfid_records.append(tag_data)

        rfid_df = pd.DataFrame(rfid_records)
        rfid_path = os.path.join(output_dir, "rfid_raw.csv")
        rfid_df.to_csv(rfid_path, index=False)

        # 保存配置文件副本
        config_path = os.path.join(output_dir, "data.yml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        print(f"Saved simulated data for {action_name} to {output_dir}")
        return {
            "kinect": kinect_path,
            "rfid": rfid_path,
            "action": action_name
        }

    def visualize_motion(self, motion_data, rfid_positions, action_id):
        """可视化生成的运动和RFID数据"""
        action_name = self.action_map[action_id]

        plt.figure(figsize=(15, 10))

        # 3D骨架轨迹可视化
        ax1 = plt.subplot(121, projection='3d')
        for j in range(self.num_tags):
            ax1.plot(
                motion_data[:, j, 0],
                motion_data[:, j, 1],
                motion_data[:, j, 2],
                label=f"Joint {j} ({self.tag_names.get(self.tag_ids[j], f'Tag {j}')})"
            )
        ax1.set_title(f"{action_name} - Kinect Joint Trajectories")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()

        # RFID相位数据可视化
        plt.subplot(122)
        for j in range(min(3, self.num_tags)):  # 只显示前3个标签
            tag_name = self.tag_names.get(self.tag_ids[j], f"Tag {j}")
            plt.plot(
                rfid_positions[:, j, 0],
                label=f"{tag_name} X"
            )
            plt.plot(
                rfid_positions[:, j, 2] - 1.0,  # 偏移以便查看
                label=f"{tag_name} Z"
            )
        plt.title(f"{action_name} - RFID Tag Positions")
        plt.xlabel("Time index")
        plt.ylabel("Position")
        plt.legend()

        # 保存可视化结果
        plt.tight_layout()
        vis_path = os.path.join(self.output_dir, f"{action_name}_motion.png")
        plt.savefig(vis_path)
        plt.close()
        print(f"Saved visualization to {vis_path}")
        return vis_path

    def generate_for_action(self, action_id):
        """生成指定动作的数据集"""
        # 1. 生成运动数据
        kinect_data = self.generate_motion(action_id)

        # 2. 模拟RFID数据采集
        rfid_data, rfid_positions = self.simulate_rfid_data(kinect_data, action_id)

        # 3. 保存数据
        data_paths = self.save_data(action_id, kinect_data, rfid_data)

        # 4. 可视化
        vis_path = self.visualize_motion(kinect_data, rfid_positions, action_id)
        data_paths["visualization"] = vis_path

        return data_paths

    def generate_all_actions(self):
        """生成所有动作类型的数据集"""
        results = {}
        for action_id in self.action_map.keys():
            print(f"Generating data for action: {self.action_map[action_id]}")
            results[self.action_map[action_id]] = self.generate_for_action(action_id)

        # 生成数据索引
        index_path = os.path.join(self.output_dir, "dataset_index.json")
        with open(index_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"All data generated and index saved to {index_path}")
        return results


# 使用示例
if __name__ == "__main__":
    # 初始化生成器
    generator = RFIDKinectDataGenerator(config_path="data.yml")

    # 生成所有动作的数据
    dataset_index = generator.generate_all_actions()

    # 打印生成的数据集信息
    print("\nGenerated datasets:")
    for action, paths in dataset_index.items():
        print(f"\nAction: {action}")
        print(f"  Kinect data: {paths['kinect']}")
        print(f"  RFID data:   {paths['rfid']}")
        print(f"  Visualization: {paths['visualization']}")

    print("\nData generation complete! All files saved to 'simulated_data' directory.")
