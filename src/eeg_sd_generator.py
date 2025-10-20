"""
EEG 驱动的 Stable Diffusion 生成工具。

本模块复用了 ``train_align.py`` 验证阶段的检索逻辑，并额外提供以下能力：

1. 将 EEG 批次映射到共享嵌入空间；
2. 从预构建的 caption 语料中检索最高相似度的文本；
3. 在文本或嵌入层面对 Top-K caption 做融合；
4. 调用 Stable Diffusion 生成图像，并按需计算质量指标。

示例::

    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader

    from dataset import EEGDataset, collate_fn_keep_captions
    from eeg_sd_generator import EEGStableDiffusionGenerator

    generator = EEGStableDiffusionGenerator(
        checkpoint_path=Path("checkpoints/MPNCELoss/xxxxxx/best_model.pt"),
    )

    dataset_cfg = generator.config["dataset"]
    val_dataset = EEGDataset(dataset_cfg, type="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_keep_captions,
    )

    batch = next(iter(val_loader))
    eeg_batch = batch["eeg_data"]
    images, retrievals, prompts = generator.generate_images_batch(
        eeg_batch,
        top_k=5,
        retain_top_n=3,
        fusion_mode="sd_embedding",
        num_images_per_prompt=1,
        batch_size=min(4, len(eeg_batch)),  # 批量大小设为4或批次大小
    )

    metrics = generator.compute_quality_metrics(
        images,
        reference_paths=batch["img_path"],
        clip_texts=prompts,
    )
    print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import logging
import numbers
import torch
import torch.nn.functional as F
from PIL import Image

from models import CLIPTextEncoder, EEGEncoder
from utils.metrics import ImageMetrics

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except ImportError as exc:  # pragma: no cover - 运行时提供友好错误信息
    raise ImportError(
        "diffusers is required for Stable Diffusion generation. "
        "Install it with `pip install diffusers`."
    ) from exc


FusionMode = Literal["text", "sd_embedding"]


@dataclass
class RetrievedCaption:
    """
    单条检索结果的封装结构。
    """

    caption: str
    similarity: float
    image_id: int
    corpus_index: int


class EEGStableDiffusionGenerator:
    """
    将 EEG 信号转化为 Stable Diffusion 图像的端到端流程。

    该类复用了 ``AlignmentTrainer`` 中的 caption 检索逻辑，保证生成阶段与
    训练/验证时的对齐策略一致。
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        config: Optional[Dict] = None,
        text_corpus_path: Optional[Path] = None,
        sd_model_name: str = "/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5",
        device: Optional[str] = None,
        sd_dtype: torch.dtype = torch.float16,
        enable_xformers: bool = False,
    ) -> None:
        """
        参数：
            checkpoint_path: EEG 编码器检查点路径。
            config: 可选配置，若为 None 则读取检查点中保存的配置。
            text_corpus_path: caption 语料库，可选覆盖配置文件路径。
            sd_model_name: Stable Diffusion 模型名称或本地目录。
            device: 运行设备，默认优先使用 CUDA。
            sd_dtype: Diffusion 管线使用的数据类型。
            enable_xformers: 是否启用内存高效注意力。
        """
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint_data.get("config")
        if config is None:
            if checkpoint_config is None:
                raise KeyError("检查点中未找到 config，请显式传入 config 参数。")
            config = checkpoint_config

        self.config = config
        self.device = torch.device(device or config.get("device", "cuda"))
        self.sd_dtype = sd_dtype

        model_cfg = config["model"]["eeg_encoder"]
        self.eeg_encoder = EEGEncoder(model_cfg).to(self.device)
        self._load_checkpoint(checkpoint_data, checkpoint_path)
        self.eeg_encoder.eval()

        text_cfg = config["model"]["text_encoder"]
        self.clip_text_encoder = CLIPTextEncoder(text_cfg["model_path"])

        corpus_path = Path(
            text_corpus_path
            or config.get("retrieval", {}).get("caption_corpus_path", "")
        )
        if not corpus_path:
            raise ValueError("caption corpus path must be provided.")
        self.text_corpus = self._build_text_corpus(corpus_path)

        # 预先通过文本投影头并做归一化，避免每次检索重复计算。
        # 语料由 CLIP 文本编码器生成，这里需要经过 EEG 编码器的文本投影头，
        # 以保证与 EEG 分支的余弦相似度对齐。
        projected = self.eeg_encoder.text_projector(
            self.text_corpus["text_vectors"].to(self.device)
        )
        self.projected_text_vectors = F.normalize(projected, dim=-1)
        self.projected_text_vectors.requires_grad_(False)

        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            torch_dtype=sd_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # 设置 DPMSolverMultistepScheduler 调度器
        self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipeline.scheduler.config, use_karras_sigmas=True
        )

        if enable_xformers and hasattr(
            self.sd_pipeline, "enable_xformers_memory_efficient_attention"
        ):
            self.sd_pipeline.enable_xformers_memory_efficient_attention()

        # 指标工具可选初始化，方便后续直接评估生成质量。
        self.image_metrics = ImageMetrics(device=str(self.device))
        self._logger = logging.getLogger(__name__)

    # --------------------------------------------------------------------- #
    # 加载相关工具函数
    # --------------------------------------------------------------------- #
    def _load_checkpoint(
        self, checkpoint_data: Dict[str, Any], checkpoint_path: Path
    ) -> None:
        """
        从训练检查点加载 EEG 编码器权重。
        """
        state_dict = checkpoint_data.get("model_state_dict")
        if state_dict is None:
            raise KeyError(
                f"Expected 'model_state_dict' in checkpoint {checkpoint_path}"
            )
        missing, unexpected = self.eeg_encoder.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Missing EEG encoder weights: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected weights in checkpoint: {unexpected}")

    def _build_text_corpus(self, path: Path) -> Dict[str, torch.Tensor]:
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

    # --------------------------------------------------------------------- #
    # 检索相关工具函数
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def encode_eeg_batch(self, eeg_batch: torch.Tensor) -> torch.Tensor:
        """
        将一批 EEG 信号投影到共享嵌入空间。

        参数：
            eeg_batch: 数据集中输出的张量，形状为 [B, channels, time]。

        返回：
            标准化后的 EEG 嵌入，形状 [B, D]。
        """
        eeg_batch = eeg_batch.to(self.device)
        embeds, _ = self.eeg_encoder(eeg_batch, return_class=False)
        return F.normalize(embeds, dim=-1)

    @torch.no_grad()
    def retrieve_captions(
        self,
        eeg_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> List[List[RetrievedCaption]]:
        """
        为每条 EEG 嵌入检索相似度最高的 captions。

        参数：
            eeg_embeddings: 归一化后的 EEG 嵌入 [B, D]。
            top_k: 每个样本召回的 caption 数量。

        返回：
            批次列表，每个元素按相似度降序排列。
        """
        eeg_embeddings = F.normalize(eeg_embeddings.to(self.device), dim=-1)
        text_vectors = self.projected_text_vectors

        top_k = min(top_k, text_vectors.size(0))
        similarity = torch.matmul(eeg_embeddings, text_vectors.t())
        values, indices = similarity.topk(k=top_k, dim=1)

        all_captions: List[List[RetrievedCaption]] = []
        for row_values, row_indices in zip(values, indices):
            results: List[RetrievedCaption] = []
            for rank, (sim, corpus_idx) in enumerate(
                zip(row_values.tolist(), row_indices.tolist())
            ):
                caption = self.text_corpus["caption"][corpus_idx]
                image_id = int(self.text_corpus["image_id"][corpus_idx].item())
                results.append(
                    RetrievedCaption(
                        caption=caption,
                        similarity=float(sim),
                        image_id=image_id,
                        corpus_index=corpus_idx,
                    )
                )
            all_captions.append(results)
        return all_captions

    # --------------------------------------------------------------------- #
    # Caption 融合工具函数
    # --------------------------------------------------------------------- #
    def _softmax_weights(self, similarities: Sequence[float]) -> torch.Tensor:
        """
        将相似度转换为概率分布，用作加权系数。
        """
        sims = torch.tensor(similarities, dtype=torch.float32, device=self.device)
        return torch.softmax(sims, dim=0)

    def _encode_sd_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        """
        将文本提示编码为 Stable Diffusion 的隐向量 [N, 77, 768]。
        """
        tokenizer = self.sd_pipeline.tokenizer
        max_length = tokenizer.model_max_length
        inputs = tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        prompt_embeds = self.sd_pipeline.text_encoder(input_ids)[0].to(
            self.sd_pipeline.unet.dtype
        )
        return prompt_embeds

    def fuse_captions(
        self,
        captions: Sequence[RetrievedCaption],
        *,
        retain_top_n: int,
        mode: FusionMode,
    ) -> Tuple[str, Optional[torch.Tensor], torch.Tensor]:
        """
        按指定策略融合 captions。

        参数：
            captions: 排好序的 ``RetrievedCaption`` 列表。
            retain_top_n: 选取前 N 条参与融合。
            mode: 融合模式，可选 "text" 或 "sd_embedding"。

        返回：
            (融合后的文本、融合后的嵌入、Softmax 权重)。
            当 ``mode="text"`` 时融合嵌入为 ``None``。
        """
        if retain_top_n <= 0:
            raise ValueError("retain_top_n must be positive.")
        if retain_top_n > len(captions):
            raise ValueError("retain_top_n exceeds available captions.")

        selected = captions[:retain_top_n]
        # 相似度来源于 [-1, 1] 的余弦得分，Softmax 能在保持可微的同时突出高置信度样本。
        weights = self._softmax_weights([cap.similarity for cap in selected])

        if mode == "text":
            fused_prompt = " ".join(cap.caption for cap in selected)
            return fused_prompt, None, weights

        if mode != "sd_embedding":
            raise ValueError(f"Unsupported fusion mode: {mode}")

        sd_embeds = self._encode_sd_prompts([cap.caption for cap in selected])
        weights_reshaped = weights.view(-1, 1, 1)
        fused_embed = torch.sum(weights_reshaped * sd_embeds, dim=0, keepdim=True)
        fused_embed = fused_embed.to(dtype=self.sd_pipeline.unet.dtype)
        fused_prompt = " ".join(cap.caption for cap in selected)
        return fused_prompt, fused_embed, weights

    # --------------------------------------------------------------------- #
    # 图像生成与评估
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate_images(
        self,
        eeg_batch: torch.Tensor,
        *,
        top_k: int = 5,
        retain_top_n: int = 3,
        fusion_mode: FusionMode = "sd_embedding",
        num_images_per_prompt: int = 1,
        negative_prompt: str = "",
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[List[Image.Image], List[List[RetrievedCaption]], List[str]]:
        """
        将一批 EEG 数据转换为图像。

        参数：
            eeg_batch: EEG 张量 [B, channels, time]。
            top_k: 检索时的候选数目。
            retain_top_n: 融合时保留的 Top-N caption。
            fusion_mode: 文本级或嵌入级的融合方式。
            num_images_per_prompt: 每个提示生成的图像数。
            negative_prompt: 可选的负向提示。
            guidance_scale: 可选的 CFG 放大系数覆盖。
            num_inference_steps: 可选的生成步数，默认使用调度器默认值。

        返回：
            生成图像列表、对应的检索结果、融合后的提示词。
        """
        eeg_embeddings = self.encode_eeg_batch(eeg_batch)
        retrievals = self.retrieve_captions(eeg_embeddings, top_k=top_k)

        all_images: List[Image.Image] = []
        fused_prompts: List[str] = []

        for captions in retrievals:
            fused_prompt, fused_embed, _ = self.fuse_captions(
                captions, retain_top_n=retain_top_n, mode=fusion_mode
            )
            fused_prompts.append(fused_prompt)

            base_guidance = (
                guidance_scale
                if guidance_scale is not None
                else getattr(self.sd_pipeline, "guidance_scale", 7.5)
            )

            # 设置生成步数
            pipeline_kwargs = {
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": base_guidance,
            }
            if num_inference_steps is not None:
                pipeline_kwargs["num_inference_steps"] = num_inference_steps

            if fused_embed is not None:
                negative = None
                if negative_prompt:
                    negative = self._encode_sd_prompts([negative_prompt])
                images = self.sd_pipeline(
                    prompt_embeds=fused_embed,
                    negative_prompt_embeds=negative,
                    **pipeline_kwargs,
                ).images
            else:
                images = self.sd_pipeline(
                    prompt=fused_prompt,
                    negative_prompt=negative_prompt or None,
                    **pipeline_kwargs,
                ).images

            all_images.extend(images)

        return all_images, retrievals, fused_prompts

    @torch.no_grad()
    def generate_images_batch(
        self,
        eeg_batch: torch.Tensor,
        *,
        top_k: int = 5,
        retain_top_n: int = 3,
        fusion_mode: FusionMode = "sd_embedding",
        num_images_per_prompt: int = 1,
        negative_prompt: str = "",
        guidance_scale: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[List[Image.Image], List[List[RetrievedCaption]], List[str]]:
        """
        批量并行生成图像的版本，提高生成效率。

        参数：
            eeg_batch: EEG 张量 [B, channels, time]。
            top_k: 检索时的候选数目。
            retain_top_n: 融合时保留的 Top-N caption。
            fusion_mode: 文本级或嵌入级的融合方式。
            num_images_per_prompt: 每个提示生成的图像数。
            negative_prompt: 可选的负向提示。
            guidance_scale: 可选的 CFG 放大系数覆盖。
            batch_size: 并行处理的批次大小，None表示使用整个批次。
            num_inference_steps: 可选的生成步数，默认使用调度器默认值。

        返回：
            生成图像列表、对应的检索结果、融合后的提示词。
        """
        eeg_embeddings = self.encode_eeg_batch(eeg_batch)
        retrievals = self.retrieve_captions(eeg_embeddings, top_k=top_k)

        # 如果没有指定批次大小，使用整个批次
        if batch_size is None:
            batch_size = len(retrievals)

        all_images: List[Image.Image] = []
        fused_prompts: List[str] = []

        # 分批处理
        for i in range(0, len(retrievals), batch_size):
            batch_retrievals = retrievals[i : i + batch_size]

            # 批量融合提示词
            batch_fused_prompts = []
            batch_fused_embeds = []

            for captions in batch_retrievals:
                fused_prompt, fused_embed, _ = self.fuse_captions(
                    captions, retain_top_n=retain_top_n, mode=fusion_mode
                )
                batch_fused_prompts.append(fused_prompt)
                if fused_embed is not None:
                    batch_fused_embeds.append(fused_embed.squeeze(0))  # 移除批次维度

            fused_prompts.extend(batch_fused_prompts)

            base_guidance = (
                guidance_scale
                if guidance_scale is not None
                else getattr(self.sd_pipeline, "guidance_scale", 7.5)
            )

            # 设置生成步数
            pipeline_kwargs = {
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": base_guidance,
            }
            if num_inference_steps is not None:
                pipeline_kwargs["num_inference_steps"] = num_inference_steps

            # 批量生成图像
            if batch_fused_embeds:
                # 堆叠嵌入向量
                stacked_embeds = torch.stack(batch_fused_embeds)

                negative = None
                if negative_prompt:
                    negative = self._encode_sd_prompts([negative_prompt])
                    # 扩展负向提示以匹配批次大小
                    negative = negative.repeat(len(batch_fused_embeds), 1, 1)

                batch_images = self.sd_pipeline(
                    prompt_embeds=stacked_embeds,
                    negative_prompt_embeds=negative,
                    **pipeline_kwargs,
                ).images
            else:
                batch_images = self.sd_pipeline(
                    prompt=batch_fused_prompts,
                    negative_prompt=negative_prompt or None,
                    **pipeline_kwargs,
                ).images

            all_images.extend(batch_images)

        return all_images, retrievals, fused_prompts

    def compute_quality_metrics(
        self,
        generated_images: Sequence[Image.Image],
        reference_paths: Sequence[str],
        *,
        clip_texts: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """
        对生成图像与参考图像进行质量评估。

        参数：
            generated_images: Stable Diffusion 输出的图像列表。
            reference_paths: 原始图像的文件路径。
            clip_texts: 兼容旧接口的占位参数（当前未使用）。

        返回：
            指标字典（FID、IS 均值与方差、CLIPScore、SSIM、LPIPS）。

        使用建议：
            - FID 对样本数量非常敏感，建议在聚合至少几十张图像后再计算；
              若仅有极少量样本，将返回 NaN 并在日志中给出提示。
            - 其它指标可以在小批量上观察趋势，但结果易受随机性影响。
        """
        if len(generated_images) != len(reference_paths):
            raise ValueError("Generated and reference image counts must match.")

        generated_images = list(generated_images)
        real_images = [Image.open(path).convert("RGB") for path in reference_paths]

        results: Dict[str, float] = {}
        if len(generated_images) >= 2:
            try:
                results["fid"] = self.image_metrics.calculate_fid(
                    generated_images, real_images
                )
            except ValueError as exc:
                self._logger.warning("Skipping FID computation: %s", exc)
                results["fid"] = float("nan")
        else:
            self._logger.info(
                "Skipping FID: need at least 2 images, received %d. "
                "Aggregate more samples for a stable estimate.",
                len(generated_images),
            )
            results["fid"] = float("nan")

        is_mean, is_std = self.image_metrics.calculate_inception_score(generated_images)
        results["is_mean"] = is_mean
        results["is_std"] = is_std

        results["clip_score"] = self.image_metrics.calculate_clip_score(
            generated_images, real_images
        )

        # 添加SSIM指标
        results["ssim"] = self.image_metrics.calculate_ssim(
            generated_images, real_images
        )

        # 添加LPIPS指标（使用AlexNet）
        results["lpips"] = self.image_metrics.calculate_lpips(
            generated_images, real_images
        )

        return results

    def visualize_samples(
        self,
        generated_images: Sequence[Image.Image],
        reference_paths: Sequence[str],
        *,
        save_path: Optional[Path] = None,
        max_samples: int = 16,
        figsize: Tuple[int, int] = (16, 8),
    ) -> None:
        """
        可视化生成图像与原始图像的对比

        参数：
            generated_images: 生成的图像列表
            reference_paths: 原始图像的文件路径列表
            save_path: 保存路径，如果为None则显示图像
            max_samples: 最大可视化样本数，默认16个（4行8列）
            figsize: 图像大小
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import random

        # 设置支持中文的字体
        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "DejaVu Sans",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        if len(generated_images) != len(reference_paths):
            raise ValueError("Generated and reference image counts must match.")

        # 随机选择样本进行可视化
        total_samples = min(len(generated_images), max_samples)
        indices = random.sample(range(len(generated_images)), total_samples)

        # 创建4行8列的子图布局
        rows, cols = 4, 8
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(
            "EEG-driven Image Generation Results\nLeft: Original, Right: Generated",
            fontsize=16,
        )

        # 确保axes是二维数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i, idx in enumerate(indices):
            row = i // (cols // 2)  # 每行显示2对图像（原始+生成）
            col_pair = (i % (cols // 2)) * 2  # 每对图像占2列

            # 加载原始图像
            original_img = Image.open(reference_paths[idx]).convert("RGB")

            # 显示原始图像（左列）
            axes[row, col_pair].imshow(original_img)
            axes[row, col_pair].set_title(f"Original {idx}", fontsize=10)
            axes[row, col_pair].axis("off")

            # 显示生成图像（右列）
            axes[row, col_pair + 1].imshow(generated_images[idx])
            axes[row, col_pair + 1].set_title(f"Generated {idx}", fontsize=10)
            axes[row, col_pair + 1].axis("off")

        # 隐藏多余的子图
        for i in range(total_samples, rows * (cols // 2)):
            row = i // (cols // 2)
            col_pair = (i % (cols // 2)) * 2
            if row < rows and col_pair < cols:
                axes[row, col_pair].axis("off")
                axes[row, col_pair + 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"可视化结果已保存到: {save_path}")

        # 始终关闭图形，不显示
        plt.close()

    def evaluate_validation_set(
        self,
        val_loader,
        *,
        max_samples: Optional[int] = None,
        visualization_samples: int = 16,
        save_dir: Optional[Path] = None,
        save_interval: int = 500,
    ) -> Dict[str, float]:
        """
        对验证集进行整体评估

        参数：
            val_loader: 验证集数据加载器
            max_samples: 最大评估样本数，None表示评估全部
            visualization_samples: 可视化样本数
            save_dir: 保存目录，如果为None则不保存文件
            save_interval: 每隔多少批次保存一次可视化图片，默认500

        返回：
            评估指标字典
        """
        import json
        import random
        from tqdm import tqdm

        # 存储所有生成的图像和对应信息
        all_generated_images = []
        all_real_images = []
        all_texts = []
        all_real_paths = []

        # 统计信息
        total_samples = len(val_loader.dataset)
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)

        print(f"开始评估验证集，总样本数: {total_samples}")

        sample_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="生成图像")):
                if sample_count >= total_samples:
                    break

                eeg_batch = batch["eeg_data"]
                real_paths = batch["img_path"]

                # 生成图像（使用批量并行方法）
                try:
                    generated_images, retrievals, prompts = self.generate_images_batch(
                        eeg_batch,
                        top_k=5,
                        retain_top_n=1,
                        fusion_mode="sd_embedding",
                        num_images_per_prompt=1,
                        batch_size=min(8, len(eeg_batch)),  # 批量大小设为8或批次大小
                        num_inference_steps=25,  # 使用DPMSolverMultistepScheduler时可以使用更少的步数
                    )

                    # 存储结果
                    all_generated_images.extend(generated_images)
                    all_real_paths.extend(real_paths)
                    all_texts.extend(prompts)

                    sample_count += len(generated_images)

                    # 每隔save_interval个批次保存一次可视化图片
                    if save_dir and batch_idx > 0 and batch_idx % save_interval == 0:
                        try:
                            # 从all_generated_images中随机选择样本进行可视化
                            current_viz_samples = min(
                                visualization_samples, len(all_generated_images)
                            )
                            if current_viz_samples > 0:
                                # 创建保存目录
                                save_dir_path = Path(save_dir)
                                save_dir_path.mkdir(parents=True, exist_ok=True)

                                # 生成带批次编号的文件名
                                viz_path = (
                                    save_dir_path
                                    / f"visualization_batch_{batch_idx}.png"
                                )

                                # 随机选择样本进行可视化
                                self.visualize_samples(
                                    generated_images=all_generated_images,
                                    reference_paths=all_real_paths[
                                        : len(all_generated_images)
                                    ],
                                    save_path=viz_path,
                                    max_samples=current_viz_samples,
                                    figsize=(16, 8),
                                )
                                print(
                                    f"批次 {batch_idx} 可视化图片已保存到: {viz_path}"
                                )
                        except Exception as e:
                            print(f"批次 {batch_idx} 可视化保存失败: {e}")

                    # 打印进度
                    if batch_idx % 10 == 0:
                        print(f"已处理 {sample_count}/{total_samples} 个样本")

                except Exception as e:
                    print(f"生成图像时出错 (批次 {batch_idx}): {e}")
                    continue

        print(f"图像生成完成，共生成 {len(all_generated_images)} 张图像")

        # 加载真实图像
        print("加载真实图像...")
        for path in tqdm(all_real_paths, desc="加载真实图像"):
            try:
                real_img = Image.open(path).convert("RGB")
                all_real_images.append(real_img)
            except Exception as e:
                print(f"加载真实图像失败 {path}: {e}")
                continue

        # 确保数量一致
        min_count = min(len(all_generated_images), len(all_real_images), len(all_texts))
        all_generated_images = all_generated_images[:min_count]
        all_real_images = all_real_images[:min_count]
        all_texts = all_texts[:min_count]

        print(f"最终评估样本数: {min_count}")

        # 计算所有指标
        print("开始计算评估指标...")
        metrics = self.compute_quality_metrics(
            generated_images=all_generated_images,
            reference_paths=all_real_paths[: len(all_generated_images)],
            clip_texts=all_texts,
        )
        metrics = {
            key: float(value) if isinstance(value, numbers.Number) else value
            for key, value in metrics.items()
        }

        # 打印指标结果
        print("=== 验证集评估结果 ===")
        for metric_name, value in metrics.items():
            if isinstance(value, numbers.Number):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")

        # 保存结果（如果指定了保存目录）
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 保存指标结果
            metrics_path = save_dir / "validation_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"评估指标已保存到: {metrics_path}")

            # 可视化随机样本
            print(f"生成可视化图像（{visualization_samples} 个样本）...")
            try:
                viz_path = save_dir / "visualization.png"
                self.visualize_samples(
                    generated_images=all_generated_images,
                    reference_paths=all_real_images[: len(all_generated_images)],
                    save_path=viz_path,
                    max_samples=visualization_samples,
                    figsize=(16, 8),
                )
            except Exception as e:
                print(f"可视化生成失败: {e}")

        return metrics


if __name__ == "__main__":
    import yaml
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader

    from dataset import EEGDataset, collate_fn_keep_captions
    from eeg_sd_generator import EEGStableDiffusionGenerator

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generator = EEGStableDiffusionGenerator(
        config=None,
        checkpoint_path=Path(
            "checkpoints/MPNCELoss/20251016_1659_mp_without_proj_head_split_by_image_id_b32_sample1/checkpoint_epoch_180.pt"
        ),
    )

    val_dataset = EEGDataset(config["dataset"], type="val")

    # 只使用验证集的0.1比例数据
    from torch.utils.data import Subset
    import random

    # 设置随机种子以确保可重现性
    random.seed(42)

    # 计算子集大小（0.1比例）
    total_size = len(val_dataset)
    subset_size = int(total_size * 0.1)

    # 随机选择索引
    indices = random.sample(range(total_size), subset_size)
    val_subset = Subset(val_dataset, indices)

    print(f"使用验证集的0.1比例数据: {subset_size}/{total_size} 样本")

    val_loader = DataLoader(
        val_subset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_keep_captions,
    )

    # 示例：对验证集进行整体评估
    print("\n=== 验证集整体评估示例 ===")
    val_metrics = generator.evaluate_validation_set(
        val_loader,
        max_samples=len(val_loader.dataset),
        visualization_samples=16,
        save_dir=Path(
            "logs/MPNCELoss/20251016_1659_mp_without_proj_head_split_by_image_id_b32_sample1"
        ),
        save_interval=10,  # 由于数据量减少，调整保存间隔为50个批次
    )
    print("验证集评估完成！")