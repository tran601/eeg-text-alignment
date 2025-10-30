import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple
import random
from datetime import datetime

from models import EEGEncoder
from models.cfm import ConditionalFlowMatching
from models.image_encoder import ImageEncoder
from dataset import EEGDataset, collate_fn_keep_captions
from utils.seed import set_seed
from PIL import Image


class LowLevelTrainer:
    """
    低级训练器 - 使用ConditionalFlowMatching从EEG生成图像嵌入
    """

    def __init__(self, config_path: str, high_level_checkpoint_path: str):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.loss_type = high_level_checkpoint_path.split("/")[1]
        self.settings = high_level_checkpoint_path.split("/")[2]

        # 设置随机种子
        set_seed(self.config["seed"])

        # 初始化日志
        self.logger = self._setup_logger()
        self.logger.info(self.config)

        # 设置设备
        self.device = torch.device(self.config["device"])

        # 初始化EEG编码器（从高级训练加载）
        self.eeg_encoder = self._load_eeg_encoder(high_level_checkpoint_path)

        # 冻结EEG编码器参数
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False
        self.eeg_encoder.eval()

        # 初始化ConditionalFlowMatching模型
        cfm_config = self.config.get("cfm", {})
        self.cfm_model = ConditionalFlowMatching(
            z_dim=cfm_config.get("z_dim", 1024),
            cond_dim=cfm_config.get("cond_dim", 512),
            hidden_dim=cfm_config.get("hidden_dim", 1024),
            n_layers=cfm_config.get("n_layers", 6),
            time_embed_dim=cfm_config.get("time_embed_dim", 128),
            dropout=cfm_config.get("dropout", 0.1),
            use_layer_norm=cfm_config.get("use_layer_norm", True),
            eps=cfm_config.get("eps", 0.0),
            topk=cfm_config.get("topk", 32),
        ).to(self.device)

        self.image_path_to_embedding, centroids_embeddings = (
            self._load_image_embeddings(
                self.config.get(
                    "image_embeddings_path",
                    "/home/chengwenjie/datasets/40classes-50images/embedding/image_embeddings.pth",
                )
            )
        )
        # 注册视觉原型
        self.cfm_model.set_visual_prototype(centroids_embeddings)

        # 注册文本嵌入
        text_corpus = self.build_text_corpus(
            Path(self.config["retrieval"]["full_caption_corpus_path"])
        )
        self.cfm_model.set_text_corpus(text_corpus)

        # 初始化图像编码器
        image_encoder_config = self.config.get("image_encoder", {})

        self.image_encoder = ImageEncoder(
            model_path=image_encoder_config.get(
                "model_path",
                "/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors",
            ),
            ip_adapter_repo=image_encoder_config.get(
                "ip_adapter_repo", "/home/chengwenjie/workspace/models/ip-adapter"
            ),
            ip_adapter_subfolder=image_encoder_config.get(
                "ip_adapter_subfolder", "models"
            ),
            ip_adapter_weight=image_encoder_config.get(
                "ip_adapter_weight", "ip-adapter_sd15.bin"
            ),
            torch_dtype=torch.float16,
            device=self.device,
        )

        # 优化器（只优化CFM模型参数）
        self.optimizer = optim.AdamW(
            self.cfm_model.parameters(),
            lr=self.config["train"]["lr"],
            weight_decay=self.config["train"]["weight_decay"],
        )

        # 混合精度训练的GradScaler
        self.scaler = GradScaler()

        # 学习率调度器
        if self.config["train"]["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["train"]["steps"]
            )
        else:
            self.scheduler = None

        # 数据加载器
        self.train_dataset = EEGDataset(config=self.config["dataset"], type="train")
        self.val_dataset = EEGDataset(config=self.config["dataset"], type="val")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn_keep_captions,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collate_fn_keep_captions,
        )

        # 检查点目录
        self.checkpoint_dir = (
            Path(self.config["checkpoint_dir"])
            / "LowLevel"
            / self.loss_type
            / self.settings
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_embeddings(self, image_embeddings_path: str):
        embeddings = torch.load(image_embeddings_path, map_location="cpu")
        image_path_to_embedding = embeddings["image_embeddings"]
        centroids_embeddings = embeddings["centroids"]
        return image_path_to_embedding, centroids_embeddings

    def _setup_logger(self) -> logging.Logger:
        """初始化并返回日志记录器"""
        logging_cfg = self.config.get("logging", {})

        log_dir = Path(logging_cfg["dir"]) / "LowLevel" / self.loss_type / self.settings
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / logging_cfg.get("filename", "train_low_level.log")

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

    def _load_eeg_encoder(self, checkpoint_path: str) -> EEGEncoder:
        """从高级训练检查点加载EEG编码器"""
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # 获取配置
        model_cfg = checkpoint_data["config"]["model"]["eeg_encoder"]

        # 创建模型
        eeg_encoder = EEGEncoder(model_cfg).to(self.device)

        # 加载权重
        state_dict = checkpoint_data["model_state_dict"]
        missing, unexpected = eeg_encoder.load_state_dict(state_dict, strict=False)

        if missing:
            self.logger.warning(f"Missing EEG encoder weights: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected weights in checkpoint: {unexpected}")

        self.logger.info(f"Successfully loaded EEG encoder from {checkpoint_path}")
        return eeg_encoder

    def _get_image_embeddings(self, image_paths: List[str]) -> torch.Tensor:

        image_embeds = []

        for image_path in image_paths:
            embeds = self.image_path_to_embedding[image_path]
            idx = torch.randint(0, len(embeds), (), device=embeds.device)
            image_embeds.append(embeds[idx])

        return torch.stack(image_embeds, dim=0)

    def train_step(self, step: int) -> Dict[str, float]:
        """训练一个步骤"""
        self.cfm_model.train()

        # 获取一个批次的数据
        batch = next(self.train_iter)
        eeg_data = batch["eeg_data"].to(self.device)
        image_path = batch["img_path"]

        # 获取EEG嵌入（条件）
        with torch.no_grad():
            eeg_embeds, _ = self.eeg_encoder(eeg_data, return_class=False)
            cond = self.cfm_model.retrieval(eeg_embeds, self.eeg_encoder, top_k=3)

        # 获取图像嵌入（目标）
        image_embeds = self._get_image_embeddings(image_path)
        image_embeds = image_embeds.to(self.device)

        # 使用混合精度训练
        self.optimizer.zero_grad()

        with autocast("cuda"):
            # 混合原型作为起点
            z_0, z_mix = self.cfm_model.mix_prototype_prior(cond)
            z_1 = image_embeds

            # 计算CFM损失，确保预测值转换为float16以匹配目标
            losses = self.cfm_model.compute_loss(z_0, z_1, cond)
            loss = losses["loss"]

        # 反向传播
        self.scaler.scale(loss).backward()

        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.cfm_model.parameters(), max_norm=1.0)

        # 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 调整学习率
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss.item(),
            "mse_loss": losses["loss_mse"].item(),
            "cosine_sim": losses["cosine_similarity"].item(),
        }

    def validate(self, step: int) -> Dict[str, float]:
        """验证并生成可视化样本"""
        self.cfm_model.eval()

        total_loss = 0
        total_mse_loss = 0
        total_cosine_sim = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                eeg_data = batch["eeg_data"].to(self.device)
                image_path = batch["img_path"]

                # 获取EEG嵌入（条件）
                eeg_embeds, _ = self.eeg_encoder(eeg_data, return_class=False)
                cond = self.cfm_model.retrieval(eeg_embeds, self.eeg_encoder, top_k=3)

                # 获取图像嵌入（目标）
                image_embeds = self._get_image_embeddings(image_path)
                image_embeds = image_embeds.to(self.device)

                # 使用混合精度计算损失
                with autocast("cuda"):
                    # 混合原型作为起点
                    z_0, z_mix = self.cfm_model.mix_prototype_prior(cond)
                    z_1 = image_embeds

                    # 计算CFM损失，确保预测值转换为float16以匹配目标
                    losses = self.cfm_model.compute_loss(z_0, z_1, cond)
                    loss = losses["loss"]

                # 记录损失
                total_loss += loss.item()
                total_mse_loss += losses["loss_mse"].item()
                total_cosine_sim += losses["cosine_similarity"].item()
                num_batches += 1

        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_cosine_sim = total_cosine_sim / num_batches
        return {
            "loss": avg_loss,
            "mse_loss": avg_mse_loss,
            "cosine_sim": avg_cosine_sim,
        }

    def _visualize_samples(
        self, original_images: List, generated_images: List, step: int
    ):
        """可视化原始图像和生成图像的对比"""
        import matplotlib.pyplot as plt
        import matplotlib

        # 设置支持中文的字体
        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "DejaVu Sans",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

        # 创建保存目录
        viz_dir = self.checkpoint_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 限制显示的样本数量
        max_samples = min(len(original_images), len(generated_images), 8)

        # 创建子图
        rows = (max_samples + 3) // 4  # 每行4个样本
        cols = min(max_samples * 2, 8)  # 每个样本显示原始和生成，最多8列

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(
            f"Step {step} - Image Generation Results\nLeft: Original, Right: Generated",
            fontsize=16,
        )

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(max_samples):
            row = i // 4
            col_pair = (i % 4) * 2

            # 显示原始图像
            if row < rows and col_pair < cols:
                axes[row, col_pair].imshow(original_images[i])
                axes[row, col_pair].set_title(f"Original {i}", fontsize=10)
                axes[row, col_pair].axis("off")

            # 显示生成图像
            if row < rows and col_pair + 1 < cols:
                axes[row, col_pair + 1].imshow(generated_images[i])
                axes[row, col_pair + 1].set_title(f"Generated {i}", fontsize=10)
                axes[row, col_pair + 1].axis("off")

        # 隐藏多余的子图
        for i in range(max_samples, rows * 4):
            row = i // 4
            col_pair = (i % 4) * 2
            if row < rows and col_pair < cols:
                axes[row, col_pair].axis("off")
            if row < rows and col_pair + 1 < cols:
                axes[row, col_pair + 1].axis("off")

        plt.tight_layout()

        # 保存图像
        save_path = viz_dir / f"step_{step}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Visualization saved to {save_path}")

    def _visualize_samples_three_columns(
        self,
        original_images: List,
        image_embed_images: List,
        eeg_embed_images: List,
        step: int,
    ):
        """可视化三列图像：原始图像、图像嵌入生成的图像、EEG嵌入生成的图像"""
        import matplotlib.pyplot as plt
        import matplotlib

        # 设置支持中文的字体
        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "DejaVu Sans",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

        # 创建保存目录
        viz_dir = self.checkpoint_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 限制显示的样本数量
        max_samples = min(
            len(original_images), len(image_embed_images), len(eeg_embed_images), 8
        )

        # 创建子图 - 每行3列（原始、图像嵌入、EEG嵌入）
        rows = (max_samples + 3) // 4  # 每行4个样本，每个样本占3列
        cols = min(max_samples * 3, 12)  # 每个样本显示3张图，最多12列

        fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
        fig.suptitle(
            f"Step {step} - Image Generation Results\nLeft: Original, Middle: Image Embed, Right: EEG Embed",
            fontsize=16,
        )

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(max_samples):
            row = i // 4
            col_triple = (i % 4) * 3

            # 显示原始图像
            if row < rows and col_triple < cols:
                axes[row, col_triple].imshow(original_images[i])
                axes[row, col_triple].set_title(f"Original {i}", fontsize=10)
                axes[row, col_triple].axis("off")

            # 显示图像嵌入生成的图像
            if row < rows and col_triple + 1 < cols:
                axes[row, col_triple + 1].imshow(image_embed_images[i])
                axes[row, col_triple + 1].set_title(f"Image Embed {i}", fontsize=10)
                axes[row, col_triple + 1].axis("off")

            # 显示EEG嵌入生成的图像
            if row < rows and col_triple + 2 < cols:
                axes[row, col_triple + 2].imshow(eeg_embed_images[i])
                axes[row, col_triple + 2].set_title(f"EEG Embed {i}", fontsize=10)
                axes[row, col_triple + 2].axis("off")

        # 隐藏多余的子图
        for i in range(max_samples, rows * 4):
            row = i // 4
            col_triple = (i % 4) * 3
            if row < rows and col_triple < cols:
                axes[row, col_triple].axis("off")
            if row < rows and col_triple + 1 < cols:
                axes[row, col_triple + 1].axis("off")
            if row < rows and col_triple + 2 < cols:
                axes[row, col_triple + 2].axis("off")

        plt.tight_layout()

        # 保存图像
        save_path = viz_dir / f"step_{step}_three_columns.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Three-column visualization saved to {save_path}")

    def save_checkpoint(self, step: int, metrics: dict):
        """保存检查点"""
        checkpoint = {
            "step": step,
            "cfm_model_state_dict": self.cfm_model.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        # 保存目标投影层（如果存在）
        if hasattr(self, "target_proj"):
            checkpoint["target_proj_state_dict"] = self.target_proj.state_dict()

        path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)

        # 保存最佳模型
        if not hasattr(self, "best_loss") or metrics["loss"] < self.best_loss:
            self.best_loss = metrics["loss"]
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        self.cfm_model.load_state_dict(checkpoint_data["cfm_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # 加载GradScaler状态
        if "scaler_state_dict" in checkpoint_data:
            self.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

        # 加载目标投影层（如果存在）
        if "target_proj_state_dict" in checkpoint_data and hasattr(self, "target_proj"):
            self.target_proj.load_state_dict(checkpoint_data["target_proj_state_dict"])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data.get("step", 0)

    def build_text_corpus(self, path: Path) -> Dict[str, torch.Tensor]:
        """
        与 ``AlignmentTrainer._build_text_corpus`` 保持一致的语料加载逻辑。
        """
        if not path.exists():
            raise FileNotFoundError(f"Caption corpus not found: {path}")
        corpus = torch.load(path, map_location="cpu")
        return {
            "image_id": corpus["image_ids"].long(),
            "caption": corpus["captions"],
            "text_vectors": corpus["embeddings"].float(),
            "image_paths": corpus.get("image_paths", []),
        }

    def generate_samples(self, step: int):
        """生成样本图像"""
        self.cfm_model.eval()

        # 收集样本用于可视化
        sample_original = []  # 原始图像
        sample_from_image_embeds = []  # 图像嵌入生成的图像
        sample_from_eeg_embeds = []  # EEG嵌入生成的图像

        with torch.no_grad():
            # 从验证集中获取一个批次
            batch = next(self.val_iter)
            eeg_data = batch["eeg_data"].to(self.device)
            image_path = batch["img_path"]

            # 获取EEG嵌入（条件）
            eeg_embeds, _ = self.eeg_encoder(eeg_data, return_class=False)
            cond = self.cfm_model.retrieval(eeg_embeds, self.eeg_encoder, top_k=3)

            # 获取图像嵌入（目标）
            image_embeds = self._get_image_embeddings(image_path)
            image_embeds = image_embeds.to(self.device)

            # 生成样本用于可视化
            num_samples = min(4, len(cond))

            # 1. 使用EEG嵌入生成图像（通过CFM模型）
            generated_embeds_from_eeg, _ = self.cfm_model.sample(
                cond[:num_samples], n_steps=100, solver="heun"
            )

            # 2. 直接使用图像嵌入生成图像（不通过CFM模型）
            for i in range(num_samples):
                # 获取原始图像
                original_img = Image.open(image_path[i]).convert("RGB")
                sample_original.append(original_img)

                # 使用图像嵌入直接生成图像
                image_gen = self.image_encoder.generate(image_embeds[i])
                sample_from_image_embeds.append(image_gen)

                # 使用EEG嵌入生成的嵌入来生成图像
                eeg_gen = self.image_encoder.generate(generated_embeds_from_eeg[i])
                sample_from_eeg_embeds.append(eeg_gen)

        # 可视化样本
        if sample_original and sample_from_image_embeds and sample_from_eeg_embeds:
            self._visualize_samples_three_columns(
                sample_original, sample_from_image_embeds, sample_from_eeg_embeds, step
            )
            self.logger.info(f"Generated {len(sample_original)} samples at step {step}")

    def train(self):
        """完整训练流程"""
        total_steps = self.config["train"]["steps"]
        generate_steps = self.config.get("generate_steps", 1000)
        val_steps = self.config.get("val_steps", 5000)

        self.logger.info(f"Starting training for {total_steps} steps")
        self.logger.info(f"Generate samples every {generate_steps} steps")
        self.logger.info(f"Validate and save checkpoint every {val_steps} steps")

        # 创建数据迭代器
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        # 训练循环
        pbar = tqdm(range(1, total_steps + 1), desc="Training")
        for step in pbar:
            try:
                # 训练一个步骤
                train_metrics = self.train_step(step)

                # 更新进度条
                pbar.set_postfix(
                    {
                        "loss": f"{train_metrics['loss']:.4f}",
                        "mse": f"{train_metrics['mse_loss']:.4f}",
                        "cosine_sim": f"{train_metrics['cosine_sim']:.4f}",
                    }
                )

                # 记录训练指标
                if step % 100 == 0:
                    self.logger.info(
                        f"Step {step} - Train: loss={train_metrics['loss']:.4f}, "
                        f"mse={train_metrics['mse_loss']:.4f}, "
                        f"cosine_sim={train_metrics['cosine_sim']:.4f}"
                    )

                # 生成样本
                if step % generate_steps == 0:
                    self.generate_samples(step)

                # 验证并保存检查点
                if step % val_steps == 0:
                    val_metrics = self.validate(step)
                    self.logger.info(f"Step {step} - Val: {val_metrics}")
                    self.save_checkpoint(step, val_metrics)

                    # 重置验证迭代器
                    self.val_iter = iter(self.val_loader)

            except StopIteration:
                # 如果数据迭代器耗尽，重新创建
                self.train_iter = iter(self.train_loader)
                continue

        # 训练结束后的最终验证和保存
        self.logger.info(
            "Training completed! Final validation and checkpoint saving..."
        )
        val_metrics = self.validate(total_steps)
        self.logger.info(f"Final validation: {val_metrics}")
        self.save_checkpoint(total_steps, val_metrics)

        self.logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--high_level_checkpoint",
        type=str,
        default="checkpoints/SupConCrossModalLoss/20251028_0911_sup_with_proj_head_512_split_by_image_id_b32_sample8_prototype/checkpoint_epoch_180.pt",
        help="Path to high-level training checkpoint",
    )
    args = parser.parse_args()

    trainer = LowLevelTrainer(args.config, args.high_level_checkpoint)
    trainer.train()
