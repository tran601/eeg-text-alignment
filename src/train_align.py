import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models import EEGEncoder, CLIPTextEncoder
from losses import MPNCELoss, SupConCrossModalLoss
from dataset import EEGDataset, collate_fn_keep_captions
from utils.seed import set_seed

from datetime import datetime


class AlignmentTrainer:
    """
    EEG-文本对齐训练器
    """

    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        loss_type = "sup" if self.config["train"]["loss"]["SupConCrossModal"] else "mp"
        proj_head = "" if self.config["model"]["eeg_encoder"]["proj_head"] else "out"
        sample_k = self.config["dataset"]["sample_k"]
        type = self.config["model"]["text_encoder"]["type"]

        self.time = (
            datetime.now().strftime("%Y%m%d_%H%M")
            + f"_{loss_type}_with{proj_head}_proj_head_split_by_image_id_{type}_sample{sample_k}"
        )

        # 设置随机种子
        set_seed(self.config["seed"])

        # 初始化日志
        self.logger = self._setup_logger()
        self.logger.info(self.config)

        # 设置设备
        self.device = torch.device(self.config["device"])

        # 初始化模型
        self.eeg_encoder = EEGEncoder(self.config["model"]["eeg_encoder"]).to(
            self.device
        )
        self.text_encoder = CLIPTextEncoder(
            self.config["model"]["text_encoder"]["model_path"]
        )

        # 初始化损失函数
        if self.config["model"]["eeg_encoder"]["class_head"]["enabled"]:
            self.class_criterion = nn.CrossEntropyLoss()
        if self.config["train"]["loss"]["SupConCrossModal"]:
            self.criterion = SupConCrossModalLoss(
                self.config["train"]["loss"]["SupConCrossModalLoss"]
            )
        else:
            self.criterion = MPNCELoss(self.config["train"]["loss"]["MPNCELoss"])
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

        # 按照图片分，每类8:2分，即训练集没有验证机图片
        self.train_dataset = EEGDataset(config=self.config["dataset"], type="train")
        self.val_dataset = EEGDataset(config=self.config["dataset"], type="val")
        self.image_id_to_name = self.train_dataset.image_ids

        # 随机分
        # dataset = EEGDataset(config=self.config["dataset"], type="all")
        # indices = torch.load(
        #     os.path.join(self.config["dataset"]["root"], "eeg_data/indices.pth")
        # )
        # self.train_dataset = Subset(dataset, indices["train"])
        # self.val_dataset = Subset(dataset, indices["eval"])
        # self.image_id_to_name = dataset.image_ids

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
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_keep_captions,
        )

        # 检查点目录
        if self.config["train"]["loss"]["SupConCrossModal"]:
            self.checkpoint_dir = (
                Path(self.config["checkpoint_dir"]) / "SupConCrossModalLoss" / self.time
            )
        else:
            self.checkpoint_dir = (
                Path(self.config["checkpoint_dir"]) / "MPNCELoss" / self.time
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 预先构建文本检索字典，便于验证阶段直接复用
        self.text_corpus = self._build_text_corpus()

    def _setup_logger(self) -> logging.Logger:
        """初始化并返回日志记录器"""
        logging_cfg = self.config.get("logging", {})

        if self.config["train"]["loss"]["SupConCrossModal"]:
            log_dir = Path(logging_cfg["dir"]) / "SupConCrossModalLoss" / self.time
        else:
            log_dir = Path(logging_cfg["dir"]) / "MPNCELoss" / self.time
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / logging_cfg.get("filename", "train_align.log")

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(
            getattr(logging, logging_cfg.get("level", "INFO").upper(), logging.INFO)
        )
        logger.propagate = False

        # 清理旧的 handler，避免重复写入
        if logger.handlers:
            logger.handlers.clear()

        log_format = logging_cfg.get("format", "%(message)s")
        formatter = logging.Formatter(log_format)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if logging_cfg.get("console", True):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        logger.debug("Logger initialized with file handler at %s", log_file)
        return logger

    def _build_text_corpus(self) -> Dict[str, Any]:
        """
        为全量数据集构建 caption 检索字典。

        返回结构:
            {
                "image_id": torch.LongTensor [N],            # 每条 caption 对应的图像 ID
                "caption": List[str] 长度为 N,               # caption 原文
                "text_vectors": torch.FloatTensor [N, D]     # caption 预先计算好的文本向量
            }
        """
        # 读取配置中提供的 caption 嵌入路径，如果没有设置则回退到默认值
        retrieval_cfg = self.config.get("retrieval", {})
        caption_embedding_path = retrieval_cfg.get(
            "caption_corpus_path",
            "/home/chengwenjie/datasets/40classes-50images/embedding/caption_embeddings.pth",
        )

        caption_embedding_path = Path(caption_embedding_path)
        if not caption_embedding_path.exists():
            raise FileNotFoundError(
                f"未找到 caption 嵌入字典文件: {caption_embedding_path}"
            )

        # 加载字典文件（包含 image_paths、image_ids、captions、embeddings 等字段）
        corpus: Dict[str, Any] = torch.load(caption_embedding_path, map_location="cpu")

        # 将嵌入向量转换为 float32，保持在 GPU 上以便快速做相似度检索，同时提前进行 L2 归一化
        text_vectors = corpus["embeddings"].float().to(self.device)

        # 图像 ID 保持 long 类型，与检索结果对齐；为了检索方便放在 GPU 上
        image_ids = corpus["image_ids"].long().to(self.device)

        # caption 和路径通常不需要搬运到 GPU，仅用于展示
        captions = corpus["captions"]

        return {
            "image_id": image_ids,
            "caption": captions,
            "text_vectors": text_vectors,
            "image_paths": corpus.get("image_paths", []),
        }

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
            eeg_embeds, class_logits = self.eeg_encoder(eeg_data, return_class=False)
            eeg_image_ids = image_id

            # 获取文本embeddings
            caps = []
            text_embeds = []
            text_image_ids = []
            for idx, cap in enumerate(caption):
                # 每个样本的多个caption
                caps.extend(cap)
                text_image_ids.append(image_id[idx].repeat(len(cap)))

            text_embeds = (
                self.text_encoder.encode_sentence(caps).float().to(self.device)
            )
            text_embeds = self.eeg_encoder.text_projector(text_embeds)
            text_embeds = F.normalize(text_embeds, dim=-1)
            text_image_ids = torch.cat(text_image_ids, dim=0)

            # 计算对齐损失
            align_loss = self.criterion(
                eeg_embeds, text_embeds, eeg_image_ids, text_image_ids
            )

            # 计算分类损失（如果启用）
            loss = align_loss
            if class_logits is not None:
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
            # torch.nn.utils.clip_grad_norm_(self.eeg_encoder.parameters(), 1.0)

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

        return {
            "loss": avg_loss,
            "align_loss": avg_align_loss,
            "class_loss": avg_class_loss,
            "accuracy": accuracy,
        }

    def validate(self, epoch: int):
        """验证"""
        self.eeg_encoder.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 收集验证集 EEG 向量及其对应的标签，用于统一进行检索评估
        all_eeg_embeds: List[torch.Tensor] = []
        all_image_ids: List[torch.Tensor] = []
        all_captions: List[List[str]] = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                eeg_data = batch["eeg_data"].to(self.device)
                caption = batch["caption"]
                image_id = batch["image_id"].to(self.device)
                class_labels = batch["class_label"].to(self.device)

                # EEG编码
                eeg_embeds, class_logits = self.eeg_encoder(
                    eeg_data, return_class=False
                )
                eeg_image_ids = image_id

                # 收集用于检索的原始数据（保留在GPU以避免重复的数据迁移）
                all_eeg_embeds.append(eeg_embeds)
                all_image_ids.append(image_id)
                all_captions.extend(caption)

                # 获取文本embeddings
                caps = []
                text_embeds = []
                text_image_ids = []
                for idx, cap in enumerate(caption):
                    caps.extend(cap)
                    text_image_ids.append(image_id[idx].repeat(len(cap)))

                text_embeds = (
                    self.text_encoder.encode_sentence(caps).float().to(self.device)
                )
                text_embeds = self.eeg_encoder.text_projector(text_embeds)
                text_embeds = F.normalize(text_embeds, dim=-1)
                text_image_ids = torch.cat(text_image_ids, dim=0)

                # 计算损失
                align_loss = self.criterion(
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

        # 将累积的张量拼接后交给检索评估逻辑
        all_eeg_embeds_tensor = torch.cat(all_eeg_embeds, dim=0)
        all_image_ids_tensor = torch.cat(all_image_ids, dim=0)

        top_k = self.config.get("retrieval", {}).get("topk", 5)
        max_examples = self.config.get("retrieval", {}).get("inspect_examples", 5)
        num_captions = self.config.get("num_captions", {}).get("num_captions", 5)
        num_captions = num_captions if num_captions <= top_k else top_k

        retrieval_metrics, retrieval_examples = self.evaluate_retrieval(
            all_eeg_embeds_tensor,
            all_image_ids_tensor,
            all_captions,
            top_k=top_k,
            max_examples=max_examples,
            num_captions=num_captions,
        )

        # 逐条打印样例，方便肉眼观察检索效果
        for example in retrieval_examples:
            self.logger.info("=" * 80)
            self.logger.info(
                "[Validation] Sample #%s | Image ID: %s | Image name: %s | Top-%s retrieval results",
                example["index"],
                example["image_id"],
                self.image_id_to_name[example["image_id"]],
                top_k,
            )
            self.logger.info("Ground-truth captions:")
            for gt_caption in example["ground_truth_captions"]:
                self.logger.info("  - %s", gt_caption)
            # 打印最相似的 caption，帮助快速比对模型预测
            top1_hit = example["topk"][0] if example["topk"] else None
            if top1_hit is not None:
                self.logger.info(
                    "Top-1 caption: sim=%.4f | caption_sim=%.4f | image_id=%s | image_name=%s | %s",
                    top1_hit["similarity"],
                    top1_hit["caption_similarity"],
                    top1_hit["image_id"],
                    self.image_id_to_name[top1_hit["image_id"]],
                    top1_hit["caption"],
                )
            self.logger.info("Retrieved captions:")
            for hit in example["topk"]:
                self.logger.info(
                    "  #%02d | sim=%.4f | caption_sim=%.4f | image_id=%s | image_name=%s | %s",
                    hit["rank"],
                    hit["similarity"],
                    hit["caption_similarity"],
                    hit["image_id"],
                    self.image_id_to_name[hit["image_id"]],
                    hit["caption"],
                )
        if retrieval_examples:
            self.logger.info("=" * 80)

        return {"loss": avg_loss, "accuracy": accuracy, **retrieval_metrics}

    @torch.inference_mode()
    def evaluate_retrieval(
        self,
        eeg_embeddings: torch.Tensor,
        image_ids: torch.Tensor,
        captions: List[List[str]],
        top_k: int = 5,
        max_examples: int = 5,
        num_captions: int = 5,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        基于预构建的文本字典执行检索评估。

        Args:
            eeg_embeddings: [N, D] 验证集 EEG 向量。
            image_ids: [N] 对应的图像 ID。
            captions: 长度为 N 的 caption 列表（每个元素为该样本的多个文本）。
            top_k: 需要检索的 caption 数量。
            max_examples: 返回用于人工检查的样例数量。

        Returns:
            (1) 检索指标字典，例如 recall@1/5/10。
            (2) 若干条用于打印的样例数据。
        """
        # 保证待评估的张量在统一设备，并进行 L2 归一化
        eeg_embeddings = F.normalize(eeg_embeddings.to(self.device), dim=-1)
        image_ids = image_ids.to(self.device)

        # 使用加载好的 caption 嵌入字典执行检索
        text_vectors = self.eeg_encoder.text_projector(self.text_corpus["text_vectors"])
        text_vectors = F.normalize(text_vectors, dim=-1)
        text_image_ids = self.text_corpus["image_id"]
        text_captions = self.text_corpus["caption"]

        # 如果配置中的 top_k 大于字典容量，需要裁剪到最大可选数量，避免运行时报错
        top_k = min(top_k, text_vectors.size(0))

        # 计算 EEG 与所有 caption 之间的相似度矩阵
        similarity = torch.matmul(eeg_embeddings, text_vectors.T)

        # 取 Top-K caption
        topk_values, topk_indices = similarity.topk(k=top_k, dim=1)
        retrieved_image_ids = text_image_ids[topk_indices]

        caption_embedding_cache: Dict[str, torch.Tensor] = {}

        def _project_captions(caption_list: List[str]) -> torch.Tensor:
            if not caption_list:
                return torch.empty(
                    0,
                    text_vectors.size(1),
                    device=self.device,
                    dtype=text_vectors.dtype,
                )

            new_captions = [
                cap for cap in caption_list if cap not in caption_embedding_cache
            ]
            if new_captions:
                encoded = self.text_encoder.encode_sentence(new_captions).to(
                    self.device, dtype=text_vectors.dtype
                )
                for cap, emb in zip(new_captions, encoded):
                    caption_embedding_cache[cap] = emb.detach()

            stacked = torch.stack(
                [caption_embedding_cache[cap] for cap in caption_list], dim=0
            )
            return F.normalize(stacked, dim=-1)

        raw_text_vectors = F.normalize(self.text_corpus["text_vectors"], dim=-1)

        caption_similarity_sum = 0.0
        caption_similarity_count = 0
        example_caption_sims: Dict[int, torch.Tensor] = {}

        for idx in range(eeg_embeddings.size(0)):
            gt_caps = captions[idx]
            retrieved_vectors = raw_text_vectors[topk_indices[idx]]
            if gt_caps:
                projected_gt = _project_captions(gt_caps)
                if projected_gt.numel() > 0 and retrieved_vectors.numel() > 0:
                    caption_sim_matrix = torch.matmul(retrieved_vectors, projected_gt.T)
                    per_hit_caption_sim = caption_sim_matrix.mean(dim=1)
                else:
                    per_hit_caption_sim = torch.zeros(
                        retrieved_vectors.size(0),
                        device=self.device,
                        dtype=text_vectors.dtype,
                    )
            else:
                per_hit_caption_sim = torch.zeros(
                    retrieved_vectors.size(0),
                    device=self.device,
                    dtype=text_vectors.dtype,
                )

            caption_similarity_sum += per_hit_caption_sim.sum().item()
            caption_similarity_count += per_hit_caption_sim.numel()
            if idx < max_examples:
                example_caption_sims[idx] = per_hit_caption_sim.detach().cpu()

        # 计算多项检索指标
        recall_metrics: Dict[str, float] = {}
        for k in (1, 5, 10, 50, 100):
            if k > top_k:
                continue
            hits_at_k = (retrieved_image_ids[:, :k] == image_ids.unsqueeze(1)).any(
                dim=1
            )
            recall_metrics[f"recall@{k}"] = hits_at_k.float().mean().item()

        if caption_similarity_count > 0:
            recall_metrics["avg_caption_similarity"] = (
                caption_similarity_sum / caption_similarity_count
            )
        else:
            recall_metrics["avg_caption_similarity"] = 0.0

        # 准备可视化样例，默认选取前 max_examples 条样本
        example_count = min(max_examples, eeg_embeddings.size(0))
        examples: List[Dict[str, Any]] = []

        for idx in range(example_count):
            hits = []
            per_hit_caption_sim = example_caption_sims.get(idx)
            for rank in range(num_captions):
                corpus_idx = topk_indices[idx, rank].item()
                # 收集 Top-K 检索出来的 caption 及其相似度，供后续打印查看
                hits.append(
                    {
                        "rank": rank + 1,
                        "image_id": text_image_ids[corpus_idx].item(),
                        "caption": text_captions[corpus_idx],
                        "similarity": topk_values[idx, rank].item(),
                        "caption_similarity": (
                            per_hit_caption_sim[rank].item()
                            if per_hit_caption_sim is not None
                            else 0.0
                        ),
                    }
                )

            examples.append(
                {
                    "index": idx,
                    "image_id": image_ids[idx].item(),
                    "ground_truth_captions": captions[idx],
                    "topk": hits,
                }
            )

        return recall_metrics, examples

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
        self.logger.info(
            "Starting training for %s epochs", self.config["train"]["epochs"]
        )

        for epoch in range(1, self.config["train"]["epochs"] + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.logger.info("Epoch %s - Train: %s", epoch, train_metrics)

            # 验证
            if epoch % self.config.get("val", 5) == 0:
                val_metrics = self.validate(epoch)
                self.logger.info("Epoch %s - Val: %s", epoch, val_metrics)

                # 保存检查点
                self.save_checkpoint(epoch, val_metrics)

            # 调整学习率
            if self.scheduler is not None:
                self.scheduler.step()

        self.logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    trainer = AlignmentTrainer(args.config)
    trainer.train()