import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import yaml
from pathlib import Path

from models import EEGEncoder, CLIPTextEncoder
from losses import MPNCELoss, SupConCrossModalLoss
from dataset import EEGDataset, collate_fn_keep_captions
from utils.metrics import calculate_metrics
from utils.seed import set_seed


class AlignmentTrainer:
    """
    EEG-文本对齐训练器
    """

    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 设置随机种子
        set_seed(self.config["seed"])

        # 设置设备
        self.device = torch.device(self.config["device"])

        # 初始化模型
        self.eeg_encoder = EEGEncoder(self.config["model"]["eeg_encoder"]).to(
            self.device
        )
        self.text_encoder = CLIPTextEncoder(
            self.config["model"]["text_encoder"]["model_name"]
        )

        # 初始化损失函数
        if self.config["model"]["class_head"]["enabled"]:
            self.class_criterion = nn.CrossEntropyLoss()
        if self.config["train"]["loss"]["SupConCrossModal"]:
            self.criterion = SupConCrossModalLoss(
                self.config["train"]["loss"]["SupConCrossModalLoss"]
            )
        else:
            self.criterion = SupConCrossModalLoss(
                self.config["train"]["loss"]["MPNCELoss"]
            )
        # 优化器
        self.optimizer = optim.AdamW(
            self.eeg_encoder.parameters(),
            lr=self.config["train"]["lr"],
            weight_decay=self.config["train"]["weight_decay"],
        )

        # 学习率调度器
        if self.config["train"]["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["train"]["epochs"]
            )
        else:
            self.scheduler = None

        # 数据加载器
        self.dataset = EEGDataset(config=self.config["dataset"], type="all")
        self.train_dataset = EEGDataset(config=self.config["dataset"], type="train")
        self.val_dataset = EEGDataset(config=self.config["dataset"], type="val")

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_keep_captions,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_keep_captions,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_keep_captions,
        )

        # 日志
        if self.config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=self.config["wandb"]["project"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )

        # 检查点目录
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.eeg_encoder.train()

        total_loss = 0
        total_align_loss = 0
        total_class_loss = 0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            eeg_data = batch["eeg_data"].to(self.device)
            caption = batch["caption"]  # List of lists
            image_id = batch["image_id"].to(self.device)
            class_labels = batch["class_label"].to(self.device)

            # EEG编码
            eeg_embeds, class_logits = self.eeg_encoder(eeg_data, return_class=True)
            eeg_image_ids = image_id

            # 获取文本embeddings
            text_embeds = []
            text_image_ids = []
            for idx, cap in enumerate(caption):
                # 每个样本的多个caption
                cap_embeds = torch.from_numpy(
                    self.text_encoder.encode_batch_sentences(cap)
                ).to(self.device)
                text_embeds.append(cap_embeds)
                text_image_ids.append(image_id[idx].repeat(cap_embeds.size(0)))
            text_embeds = torch.cat(text_embeds, dim=0)
            text_image_ids = torch.cat(text_image_ids, dim=0)

            # 计算对齐损失
            align_loss = self.criterion(
                eeg_embeds, text_embeds, eeg_image_ids, text_image_ids
            )

            # 计算分类损失（如果启用）
            loss = align_loss
            if (
                self.config["model"]["class_head"]["enabled"]
                and class_logits is not None
            ):
                class_loss = self.class_criterion(class_logits, class_labels)
                loss = (
                    align_loss + self.config["train"]["class_loss_weight"] * class_loss
                )

                # 统计分类准确率
                _, predicted = torch.max(class_logits, 1)
                correct_predictions += (predicted == class_labels).sum().item()
                total_samples += class_labels.size(0)
                total_class_loss += class_loss.item()

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.eeg_encoder.parameters(), 1.0)

            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_align_loss += align_loss.item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "align": f"{align_loss.item():.4f}",
                    "acc": f"{correct_predictions/max(total_samples, 1):.2%}",
                }
            )

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_align_loss = total_align_loss / len(self.train_loader)
        avg_class_loss = (
            total_class_loss / len(self.train_loader) if total_samples > 0 else 0
        )
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # 记录到wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/align_loss": avg_align_loss,
                    "train/class_loss": avg_class_loss,
                    "train/accuracy": accuracy,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

        return {
            "loss": avg_loss,
            "align_loss": avg_align_loss,
            "class_loss": avg_class_loss,
            "accuracy": accuracy,
        }

    def validate(self, epoch: int):
        """验证"""
        self.eeg_encoder.eval()

        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # 收集embeddings用于检索评估
        all_eeg_embeds = []
        all_text_embeds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                eeg_data = batch["eeg_data"].to(self.device)
                caption = batch["caption"]
                image_id = batch["image_id"].to(self.device)
                class_labels = batch["class_label"].to(self.device)

                # EEG编码
                eeg_embeds, class_logits = self.eeg_encoder(eeg_data, return_class=True)
                eeg_image_ids = image_id

                # 收集embeddings
                all_eeg_embeds.append(eeg_embeds.cpu().numpy())
                all_labels.append(class_labels.cpu().numpy())

                # 获取文本embeddings
                text_embeds = []
                text_image_ids = []
                for idx, cap in enumerate(caption):
                    cap_embeds = torch.from_numpy(
                        self.text_encoder.encode_batch_sentences(cap)
                    ).to(self.device)
                    text_embeds.append(cap_embeds)
                    text_image_ids.append(image_id[idx].repeat(cap_embeds.szie(0)))
                text_embeds = torch.cat(text_embeds, dim=0)
                text_image_ids = torch.cat(text_image_ids, dim=0)

                # 计算损失
                align_loss = self.mp_infonce(
                    eeg_embeds, text_embeds, eeg_image_ids, text_image_ids
                )
                total_loss += align_loss.item()

                # 分类准确率
                if class_logits is not None:
                    _, predicted = torch.max(class_logits, 1)
                    correct_predictions += (predicted == class_labels).sum().item()
                    total_samples += class_labels.size(0)

        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # 计算检索指标
        all_eeg_embeds = np.concatenate(all_eeg_embeds)
        all_labels = np.concatenate(all_labels)

        # 简单的检索评估：计算同类样本的平均排名
        retrieval_metrics = self.evaluate_retrieval(all_eeg_embeds, all_labels)

        # 记录到wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "val/loss": avg_loss,
                    "val/accuracy": accuracy,
                    "val/recall@1": retrieval_metrics["recall@1"],
                    "val/recall@5": retrieval_metrics["recall@5"],
                    "epoch": epoch,
                }
            )

        return {"loss": avg_loss, "accuracy": accuracy, **retrieval_metrics}

    def evaluate_retrieval(self, embeddings: np.ndarray, labels: np.ndarray):
        """评估检索性能"""
        n = len(embeddings)

        # 计算相似度矩阵
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim_matrix = embeddings @ embeddings.T

        # 设置对角线为-inf（排除自己）
        np.fill_diagonal(sim_matrix, -np.inf)

        # 计算recall@k
        recall_at_k = {}
        for k in [1, 5, 10]:
            correct = 0
            for i in range(n):
                # 找到top-k最相似的样本
                top_k_idx = np.argpartition(sim_matrix[i], -k)[-k:]
                top_k_labels = labels[top_k_idx]

                # 检查是否有同类样本
                if labels[i] in top_k_labels:
                    correct += 1

            recall_at_k[f"recall@{k}"] = correct / n

        return recall_at_k

    def save_checkpoint(self, epoch: int, metrics: dict):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.eeg_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # 保存最佳模型
        if (
            not hasattr(self, "best_accuracy")
            or metrics["accuracy"] > self.best_accuracy
        ):
            self.best_accuracy = metrics["accuracy"]
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def train(self):
        """完整训练流程"""
        print(f"Starting training for {self.config['train']['epochs']} epochs")

        for epoch in range(1, self.config["train"]["epochs"] + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train: {train_metrics}")

            # 验证
            if epoch % self.config.get("val_interval", 5) == 0:
                val_metrics = self.validate(epoch)
                print(f"Epoch {epoch} - Val: {val_metrics}")

                # 保存检查点
                self.save_checkpoint(epoch, val_metrics)

            # 调整学习率
            if self.scheduler is not None:
                self.scheduler.step()

        print("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    trainer = AlignmentTrainer(args.config)
    trainer.train()