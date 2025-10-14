import os

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from scipy import signal
import json
import random


def collate_fn_keep_captions(batch):
    # 张量/数值用默认逻辑堆叠
    eeg_data = default_collate([b["eeg_data"] for b in batch])  # [B, ...]
    class_label = default_collate([b["class_label"] for b in batch])  # [B]
    image_id = default_collate([b["image_id"] for b in batch])  # [B]
    img_path = [b["img_path"] for b in batch]  # list[str], 长度 B

    # 关键：caption 不再让 default_collate 处理，保持为 list[list[str]]，形状 [B, K]
    captions = [b["caption"] for b in batch]  # [[str]*K]*B

    return {
        "eeg_data": eeg_data,
        "class_label": class_label,
        "image_id": image_id,
        "img_path": img_path,
        "caption": captions,  # [B, K]
    }


class EEGDataset(Dataset):
    def __init__(self, config, type="all"):
        root = config["root"]
        if type == "all":
            data_name = config["all_data"]
        elif type == "train":
            data_name = config["train_data"]
        elif type == "val":
            data_name = config["val_data"]
        else:
            raise ValueError(f"type value is {type}, not in ['all','train','val']")
        caption_name = config["caption_name"]
        sample_k = config["sample_k"]
        self.root = root
        self.sample_k = sample_k
        self.caption_path = os.path.join(root, "caption", caption_name)

        with open(self.caption_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)

        self.data_path = os.path.join(root, "eeg_data", data_name)
        data = torch.load(self.data_path, map_location="cpu")
        self.dataset = data["dataset"]
        self.labels = data["labels"]
        self.image_ids = data["images"]

    def __len__(self):
        return len(self.dataset)

    def filter(self, eeg_data, HZ, low_f, high_f):
        b, a = signal.butter(2, [low_f * 2 / HZ, high_f * 2 / HZ], "bandpass")
        eeg_data = signal.lfilter(b, a, eeg_data).copy()
        eeg_data = torch.from_numpy(eeg_data).float()
        return eeg_data

    def __getitem__(self, idx, HZ=1000):
        data_item = self.dataset[idx]
        eeg_data = data_item["eeg_data"][:, 20:460]
        eeg_data = self.filter(eeg_data, HZ, 1, 70)

        label = self.labels.index(data_item["label"])
        image_id = self.image_ids.index(os.path.splitext(data_item["image"])[0])

        img_name = data_item["image"]
        image_path = os.path.join(self.root, "images", data_item["label"], img_name)
        caption = random.sample(self.captions[image_path], self.sample_k)

        return {
            "eeg_data": eeg_data,
            "caption": caption,
            "class_label": label,
            "image_id": image_id,
            "img_path": image_path,
        }


from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Subset

if __name__ == "__main__":
    config = {
        "root": "/home/chengwenjie/datasets/40classes-50images",
        "data_name": "dataset.pth",
        "caption_name": "qwen3vl_multi_caption.json",
        "sample_k": 4,
    }
    dataset = EEGDataset(config)
    indices = torch.load(
        os.path.join(
            "/home/chengwenjie/datasets/40classes-50images", "eeg_data/indices.pth"
        ),
        map_location="cpu",
    )
    train_dataset = Subset(dataset, indices["train"])
    eval_dataset = Subset(dataset, indices["eval"])
    dataloader = DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=collate_fn_keep_captions,
    )
    for item in tqdm(dataloader):
        pass