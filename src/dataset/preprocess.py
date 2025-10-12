import os
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from sklearn.model_selection import train_test_split
import random
import yaml

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    数据预处理器
    负责处理原始EEG数据并构建索引
    """

    def __init__(self, config: Dict):
        """
        初始化预处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.data_dir = Path(config["data"]["root"])
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # EEG配置
        self.eeg_config = config["data"]["eeg"]
        self.fs = self.eeg_config["fs"]
        self.channels = self.eeg_config.get("channels", 128)
        self.window_sec = self.eeg_config.get("window_sec", 1.0)
        self.step_sec = self.eeg_config.get("step_sec", 0.5)
        self.window_samples = int(self.window_sec * self.fs)
        self.step_samples = int(self.step_sec * self.fs)

        # 预处理参数
        self.bandpass = self.eeg_config.get("bandpass", [0.5, 70])
        self.notch = self.eeg_config.get("notch", 50)
        self.norm_method = self.eeg_config.get("norm", "channel_zscore")

        # 分割配置
        self.split_config = config["data"]["splits"]

    def prepare_all(self):
        """执行所有预处理步骤"""
        logger.info("Starting data preprocessing...")

        # 1. 处理EEG数据
        self._process_eeg_data()

        # 2. 处理文本数据
        self._process_text_data()

        # 3. 创建数据集分割
        self._create_splits()

        logger.info("Data preprocessing completed!")

    def _process_eeg_data(self):
        """处理EEG数据"""
        logger.info("Processing EEG data...")

        # 原始EEG数据目录
        raw_eeg_dir = self.data_dir / "raw" / "eeg"
        processed_eeg_dir = self.processed_dir / "eeg"
        processed_eeg_dir.mkdir(exist_ok=True)

        # 获取所有EEG文件
        eeg_files = list(raw_eeg_dir.glob("*.npy")) + list(raw_eeg_dir.glob("*.pkl"))

        logger.info(f"Found {len(eeg_files)} EEG files")

        # 处理每个文件
        for eeg_file in eeg_files:
            logger.info(f"Processing {eeg_file.name}")

            # 加载EEG数据
            if eeg_file.suffix == ".npy":
                eeg_data = np.load(eeg_file)
            else:
                with open(eeg_file, "rb") as f:
                    eeg_data = pickle.load(f)

            # 预处理
            eeg_data = self._preprocess_eeg(eeg_data)

            # 分割窗口
            windows = self._segment_eeg(eeg_data)

            # 保存处理后的数据
            base_name = eeg_file.stem
            for i, window in enumerate(windows):
                output_path = processed_eeg_dir / f"{base_name}_win_{i:04d}.npy"
                np.save(output_path, window)

        logger.info(f"Processed EEG data saved to {processed_eeg_dir}")

    def _process_text_data(self):
        """处理文本数据"""
        logger.info("Processing text data...")

        # 加载原始文本数据
        raw_text_path = self.data_dir / "raw" / "captions.json"

        if not raw_text_path.exists():
            logger.warning(f"Raw text file not found: {raw_text_path}")
            # 创建示例数据
            self._create_example_text_data()
            raw_text_path = self.data_dir / "raw" / "captions.json"

        with open(raw_text_path, "r") as f:
            raw_captions = json.load(f)

        # 处理文本数据
        processed_captions = []
        for item in raw_captions:
            processed_item = {"image_id": item["image_id"], "captions": []}

            for caption in item["captions"]:
                processed_caption = {
                    "id": len(processed_captions),
                    "text": caption["text"],
                    "class_id": caption.get("class_id", 0),
                }
                processed_item["captions"].append(processed_caption)

            processed_captions.append(processed_item)

        # 保存处理后的文本数据
        text_output_path = self.processed_dir / "captions.json"
        with open(text_output_path, "w") as f:
            json.dump(processed_captions, f, indent=2)

        logger.info(f"Processed text data saved to {text_output_path}")

    def _create_example_text_data(self):
        """创建示例文本数据"""
        logger.info("Creating example text data...")

        # 创建示例类别
        classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        # 创建示例captions
        example_captions = []
        for i in range(100):  # 100张示例图像
            class_id = i % len(classes)
            class_name = classes[class_id]

            # 为每张图像创建多个caption
            captions = [
                f"a photo of a {class_name}",
                f"a picture showing a {class_name}",
                f"this is a {class_name}",
                f"an image of a {class_name}",
            ]

            example_captions.append(
                {
                    "image_id": i,
                    "captions": [
                        {"text": caption, "class_id": class_id} for caption in captions
                    ],
                }
            )

        # 保存示例数据
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        with open(raw_dir / "captions.json", "w") as f:
            json.dump(example_captions, f, indent=2)

        logger.info(f"Example text data saved to {raw_dir / 'captions.json'}")

    def _create_splits(self):
        """创建数据集分割"""
        logger.info("Creating dataset splits...")

        # 获取所有处理后的EEG文件
        processed_eeg_dir = self.processed_dir / "eeg"
        eeg_files = list(processed_eeg_dir.glob("*.npy"))

        # 按基础文件名分组（同一文件的窗口属于同一组）
        file_groups = {}
        for eeg_file in eeg_files:
            base_name = "_".join(eeg_file.stem.split("_")[:-2])  # 提取基础名称
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(eeg_file)

        # 获取基础文件列表
        base_files = list(file_groups.keys())

        # 随机打乱
        random.shuffle(base_files)

        # 计算分割点
        n_files = len(base_files)
        train_end = int(n_files * self.split_config["train_ratio"])
        val_end = train_end + int(n_files * self.split_config["val_ratio"])

        # 分割
        train_files = base_files[:train_end]
        val_files = base_files[train_end:val_end]
        test_files = base_files[val_end:]

        # 创建分割索引
        splits = {
            "train": self._create_split_index(train_files, file_groups),
            "val": self._create_split_index(val_files, file_groups),
            "test": self._create_split_index(test_files, file_groups),
        }

        # 保存分割索引
        for split_name, split_data in splits.items():
            split_path = self.processed_dir / f"{split_name}_index.json"
            with open(split_path, "w") as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"{split_name} split saved to {split_path}")

    def _create_split_index(
        self, base_files: List[str], file_groups: Dict
    ) -> List[Dict]:
        """创建单个分割的索引"""
        # 加载文本数据
        text_path = self.processed_dir / "captions.json"
        with open(text_path, "r") as f:
            captions_data = json.load(f)

        # 创建图像ID到文本的映射
        image_to_captions = {}
        for item in captions_data:
            image_to_captions[item["image_id"]] = item["captions"]

        split_index = []

        for base_file in base_files:
            # 获取该文件的所有窗口
            eeg_files = file_groups[base_file]

            # 选择一个窗口作为代表（简化处理）
            eeg_file = eeg_files[0]

            # 提取图像ID（假设文件名包含图像ID）
            try:
                image_id = int(base_file.split("_")[-1])
            except (ValueError, IndexError):
                # 如果无法提取，使用随机ID
                image_id = hash(base_file) % 1000

            # 获取对应的captions
            captions = image_to_captions.get(image_id, [])

            if not captions:
                # 如果没有找到对应的captions，使用默认的
                class_id = image_id % 10  # 假设有10个类别
                default_captions = [
                    {
                        "ids": [0, 1, 2, 3],  # 示例ID
                        "class_id": class_id,
                        "image_id": image_id,
                    }
                ]
                captions = default_captions

            # 添加到索引
            split_index.append(
                {
                    "eeg_path": f"eeg/{eeg_file.name}",
                    "image_path": f"images/image_{image_id:04d}.jpg",
                    "captions": captions,
                }
            )

        return split_index

    def _preprocess_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """
        预处理EEG数据

        Args:
            eeg: 原始EEG数据 (channels, timepoints)

        Returns:
            预处理后的EEG数据
        """
        # 带通滤波
        if self.bandpass:
            low, high = self.bandpass
            nyquist = self.fs / 2
            low_norm = low / nyquist
            high_norm = high / nyquist

            b, a = signal.butter(4, [low_norm, high_norm], btype="band")
            eeg = signal.filtfilt(b, a, eeg, axis=-1)

        # 陷波滤波（去除工频干扰）
        if self.notch:
            nyquist = self.fs / 2
            notch_norm = self.notch / nyquist
            b, a = signal.iirnotch(notch_norm, 30)
            eeg = signal.filtfilt(b, a, eeg, axis=-1)

        # 标准化
        if self.norm_method == "channel_zscore":
            mean = np.mean(eeg, axis=-1, keepdims=True)
            std = np.std(eeg, axis=-1, keepdims=True)
            eeg = (eeg - mean) / (std + 1e-8)
        elif self.norm_method == "global_zscore":
            mean = np.mean(eeg)
            std = np.std(eeg)
            eeg = (eeg - mean) / (std + 1e-8)
        elif self.norm_method == "minmax":
            eeg_min = np.min(eeg, axis=-1, keepdims=True)
            eeg_max = np.max(eeg, axis=-1, keepdims=True)
            eeg = (eeg - eeg_min) / (eeg_max - eeg_min + 1e-8)

        return eeg.astype(np.float32)

    def _segment_eeg(self, eeg: np.ndarray) -> List[np.ndarray]:
        """
        将EEG数据分割成窗口

        Args:
            eeg: EEG数据 (channels, timepoints)

        Returns:
            分割后的EEG窗口列表
        """
        n_channels, n_timepoints = eeg.shape

        # 计算窗口数量
        n_windows = (n_timepoints - self.window_samples) // self.step_samples + 1

        if n_windows <= 0:
            # 如果EEG长度小于窗口大小，进行零填充
            pad_width = self.window_samples - n_timepoints
            eeg = np.pad(eeg, ((0, 0), (0, pad_width)), mode="constant")
            return [eeg]

        # 分割
        windows = []
        for i in range(n_windows):
            start = i * self.step_samples
            end = start + self.window_samples
            window = eeg[:, start:end]
            windows.append(window)

        return windows