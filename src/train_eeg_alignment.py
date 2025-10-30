import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from PIL import Image

from models import EEGEncoder
from models.image_encoder import ImageEncoder
from dataset import EEGDataset, collate_fn_keep_captions
from utils.seed import set_seed


class EEGAlignmentTrainer:
    """
    直接对齐EEG嵌入与图像嵌入的训练器。
    使用MSE与余弦损失联合优化EEG编码器，使其输出与图像嵌入一致。
    """

    def __init__(self, config_path: str, run_name: Optional[str] = None):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 运行名称，用于日志与检查点路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or timestamp
        self.loss_type = "MSECosine"
        self.settings = self.run_name

        # 设置随机种子
        set_seed(self.config["seed"])

        # 设备
        self.device = torch.device(self.config["device"])

        # 初始化日志
        self.logger = self._setup_logger()
        self.logger.info(self.config)

        # 数据集与数据加载器
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

        # 初始化EEG编码器
        eeg_cfg = self.config["model"]["eeg_encoder"]
        eeg_cfg["proj_head_dim"] = 1024
        self.eeg_encoder = EEGEncoder(eeg_cfg).to(self.device)

        # 图像嵌入
        (
            self.image_path_to_embedding,
            self.centroids_embeddings,
        ) = self._load_image_embeddings(
            self.config.get(
                "image_embeddings_path",
                "/home/chengwenjie/datasets/40classes-50images/embedding/image_embeddings.pth",
            )
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
                self.optimizer, T_max=self.config["train"]["steps"]
            )
        else:
            self.scheduler = None

        # 图像编码器，用于验证与可视化
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
            device=self.device.type,
        )

        # 检查点目录
        self.checkpoint_dir = (
            Path(self.config["checkpoint_dir"])
            / "EEGAlignment"
            / self.loss_type
            / self.settings
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_embeddings(self, image_embeddings_path: str):
        embeddings = torch.load(image_embeddings_path, map_location="cpu")
        image_path_to_embedding = embeddings["image_embeddings"]
        centroids_embeddings = embeddings.get("centroids")
        return image_path_to_embedding, centroids_embeddings

    def _setup_logger(self) -> logging.Logger:
        logging_cfg = self.config.get("logging", {})
        log_dir = (
            Path(logging_cfg["dir"]) / "EEGAlignment" / self.loss_type / self.settings
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / logging_cfg.get("filename", "train_eeg_alignment.log")

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(
            getattr(logging, logging_cfg.get("level", "INFO").upper(), logging.INFO)
        )
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(logging_cfg.get("format", "%(message)s"))

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if logging_cfg.get("console", True):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def _get_image_embeddings(self, image_paths: List[str]) -> torch.Tensor:
        image_embeds = []
        for image_path in image_paths:
            embeds = self.image_path_to_embedding[image_path]
            if embeds.ndim == 1:
                image_embeds.append(embeds)
            else:
                image_embeds.append(embeds[0])
        return torch.stack(image_embeds, dim=0)

    def _forward_alignment(self, eeg_data: torch.Tensor) -> torch.Tensor:
        eeg_embeds, _ = self.eeg_encoder(eeg_data, return_class=False)
        return eeg_embeds

    def train_step(self, step: int) -> Dict[str, float]:
        self.eeg_encoder.train()

        batch = next(self.train_iter)
        eeg_data = batch["eeg_data"].to(self.device, non_blocking=True)
        image_paths = batch["img_path"]

        image_embeds = self._get_image_embeddings(image_paths).to(
            self.device, dtype=torch.float32, non_blocking=True
        )

        self.optimizer.zero_grad(set_to_none=True)

        aligned_eeg = self._forward_alignment(eeg_data)
        mse_loss = F.mse_loss(aligned_eeg, image_embeds)
        cosine_sim = F.cosine_similarity(aligned_eeg, image_embeds, dim=-1)
        cosine_loss = 1.0 - cosine_sim.mean()
        loss = mse_loss + cosine_loss

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eeg_encoder.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "cosine_sim": cosine_sim.mean().item(),
        }

    def validate(self, step: int) -> Dict[str, float]:
        self.eeg_encoder.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_cosine_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                eeg_data = batch["eeg_data"].to(self.device, non_blocking=True)
                image_paths = batch["img_path"]
                image_embeds = self._get_image_embeddings(image_paths).to(
                    self.device, dtype=torch.float32, non_blocking=True
                )

                aligned_eeg = self._forward_alignment(eeg_data)
                mse_loss = F.mse_loss(aligned_eeg, image_embeds)
                cosine_sim = F.cosine_similarity(aligned_eeg, image_embeds, dim=-1)
                cosine_loss = 1.0 - cosine_sim.mean()
                loss = mse_loss + cosine_loss

                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_cosine_loss += cosine_loss.item()
                total_cosine_sim += cosine_sim.mean().item()
                num_batches += 1

        if num_batches == 0:
            return {"loss": 0.0, "mse_loss": 0.0, "cosine_loss": 0.0, "cosine_sim": 0.0}

        return {
            "loss": total_loss / num_batches,
            "mse_loss": total_mse / num_batches,
            "cosine_loss": total_cosine_loss / num_batches,
            "cosine_sim": total_cosine_sim / num_batches,
        }

    def _visualize_samples_three_columns(
        self,
        original_images: List[Image.Image],
        image_embed_images: List[Image.Image],
        eeg_embed_images: List[Image.Image],
        step: int,
    ):
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "DejaVu Sans",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

        viz_dir = self.checkpoint_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        max_samples = min(
            len(original_images), len(image_embed_images), len(eeg_embed_images), 8
        )
        rows = (max_samples + 3) // 4
        cols = min(max_samples * 3, 12)

        fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
        fig.suptitle(
            f"Step {step} - Alignment Results\nLeft: Original, Middle: Image Embedding, Right: EEG Embedding",
            fontsize=16,
        )

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(max_samples):
            row = i // 4
            col_triple = (i % 4) * 3

            axes[row, col_triple].imshow(original_images[i])
            axes[row, col_triple].set_title(f"Original {i}", fontsize=10)
            axes[row, col_triple].axis("off")

            axes[row, col_triple + 1].imshow(image_embed_images[i])
            axes[row, col_triple + 1].set_title(f"Image Embed {i}", fontsize=10)
            axes[row, col_triple + 1].axis("off")

            axes[row, col_triple + 2].imshow(eeg_embed_images[i])
            axes[row, col_triple + 2].set_title(f"EEG Embed {i}", fontsize=10)
            axes[row, col_triple + 2].axis("off")

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
        save_path = viz_dir / f"step_{step}_alignment.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Visualization saved to {save_path}")

    def generate_samples(self, step: int):
        self.eeg_encoder.eval()

        try:
            batch = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            batch = next(self.val_iter)

        eeg_data = batch["eeg_data"].to(self.device, non_blocking=True)
        image_paths = batch["img_path"]

        image_embeds = self._get_image_embeddings(image_paths).to(
            self.device, non_blocking=True
        )

        with torch.no_grad():
            aligned_eeg = self._forward_alignment(eeg_data)

        sample_original = []
        sample_from_image_embeds = []
        sample_from_eeg_embeds = []

        num_samples = min(4, aligned_eeg.size(0))

        for i in range(num_samples):
            original_img = Image.open(image_paths[i]).convert("RGB")
            sample_original.append(original_img)

            image_gen = self.image_encoder.generate(image_embeds[i].detach().cpu())
            sample_from_image_embeds.append(image_gen)

            eeg_gen = self.image_encoder.generate(aligned_eeg[i].detach().cpu())
            sample_from_eeg_embeds.append(eeg_gen)

        if sample_original and sample_from_image_embeds and sample_from_eeg_embeds:
            self._visualize_samples_three_columns(
                sample_original, sample_from_image_embeds, sample_from_eeg_embeds, step
            )
            self.logger.info(f"Generated {len(sample_original)} samples at step {step}")

    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "config": self.config,
            "eeg_encoder_state_dict": self.eeg_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)

        if not hasattr(self, "best_loss") or metrics["loss"] < self.best_loss:
            self.best_loss = metrics["loss"]
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        self.eeg_encoder.load_state_dict(checkpoint_data["eeg_encoder_state_dict"])

        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data.get("step", 0)

    def train(self, start_step: int = 0):
        total_steps = self.config["train"]["steps"]
        generate_steps = self.config.get("generate_steps", 1000)
        val_steps = self.config.get("val_steps", 5000)

        self.logger.info(
            f"Starting training for {total_steps} steps (resume from {start_step})"
        )
        self.logger.info(f"Generate samples every {generate_steps} steps")
        self.logger.info(f"Validate and save checkpoint every {val_steps} steps")

        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        pbar = tqdm(
            range(start_step + 1, total_steps + 1),
            desc="Training",
            initial=start_step,
            total=total_steps,
        )
        for step in pbar:
            try:
                train_metrics = self.train_step(step)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                train_metrics = self.train_step(step)

            pbar.set_postfix(
                {
                    "loss": f"{train_metrics['loss']:.4f}",
                    "mse_loss": f"{train_metrics['mse_loss']:.4f}",
                    "cos_loss": f"{train_metrics['cosine_loss']:.4f}",
                }
            )

            if step % 100 == 0:
                self.logger.info(
                    f"Step {step} - Train: loss={train_metrics['loss']:.4f}, "
                    f"mse={train_metrics['mse_loss']:.4f}, "
                    f"cosine_loss={train_metrics['cosine_loss']:.4f}, "
                    f"cosine_sim={train_metrics['cosine_sim']:.4f}"
                )

            if step % generate_steps == 0:
                self.generate_samples(step)

            if step % val_steps == 0:
                val_metrics = self.validate(step)
                self.logger.info(f"Step {step} - Val: {val_metrics}")
                self.save_checkpoint(step, val_metrics)
                self.val_iter = iter(self.val_loader)

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
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()

    trainer = EEGAlignmentTrainer(args.config, run_name=args.run_name)
    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = trainer.load_checkpoint(args.resume_checkpoint)
    trainer.train(start_step=start_step)
