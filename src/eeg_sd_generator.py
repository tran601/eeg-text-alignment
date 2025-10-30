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
from models.cfm import ConditionalFlowMatching
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
        cfm_checkpoint_path: Optional[Path] = None,
        use_ip_adapter: bool = True,
        ip_adapter_scale: float = 0.2,
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
            cfm_checkpoint_path: CFM 模型检查点路径。
            use_ip_adapter: 是否使用 IP-Adapter。
            ip_adapter_scale: IP-Adapter 强度。
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
            or config.get("retrieval", {}).get("full_caption_corpus_path", "")
        )
        if not corpus_path:
            raise ValueError("caption corpus path must be provided.")
        self.text_corpus = self._build_text_corpus(corpus_path)

        # 预先通过文本投影头并做归一化，避免每次检索重复计算。
        # 语料由 CLIP 文本编码器生成，这里需要经过 EEG 编码器的文本投影头，
        # 以保证与 EEG 分支的余弦相似度对齐。
        with torch.no_grad():
            projected = self.eeg_encoder.text_projector(
                self.text_corpus["text_vectors"].to(self.device)
            )
            self.projected_text_vectors = F.normalize(projected, dim=-1)

        self.sd_pipeline = StableDiffusionPipeline.from_single_file(
            sd_model_name, torch.dtype=torch.float16, variant="fp16"
        ).to(self.device)

        # 设置 DPMSolverMultistepScheduler 调度器
        self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipeline.scheduler.config, use_karras_sigmas=True
        )

        if enable_xformers and hasattr(
            self.sd_pipeline, "enable_xformers_memory_efficient_attention"
        ):
            self.sd_pipeline.enable_xformers_memory_efficient_attention()

        # 初始化 IP-Adapter
        self.use_ip_adapter = use_ip_adapter
        self.ip_adapter_scale = ip_adapter_scale
        if self.use_ip_adapter:
            self.sd_pipeline.load_ip_adapter(
                "/home/chengwenjie/workspace/models/ip-adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            self.sd_pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

        # 初始化 CFM 模型
        cfm_checkpoint = torch.load(cfm_checkpoint_path, map_location="cpu")
        self.cfm_model = None
        if cfm_checkpoint_path is not None:
            cfm_config = cfm_checkpoint["config"].get("cfm", {})
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
            self.cfm_model.set_text_corpus(self.text_corpus)

            # 加载 CFM 检查点
            if "cfm_model_state_dict" in cfm_checkpoint:
                self.cfm_model.load_state_dict(cfm_checkpoint["cfm_model_state_dict"])
            else:
                self.cfm_model.load_state_dict(cfm_checkpoint)
            self.cfm_model.eval()

        # 指标工具可选初始化，方便后续直接评估生成质量。
        self.image_metrics = ImageMetrics(device=str(self.device))

    # --------------------------------------------------------------------- #
    # 加载相关工具函数
    # --------------------------------------------------------------------- #

    def _load_image_embeddings(self, image_embeddings_path: str):
        embeddings = torch.load(image_embeddings_path, map_location="cpu")
        image_path_to_embedding = embeddings["image_embeddings"]
        centroids_embeddings = embeddings["centroids"]
        return image_path_to_embedding, centroids_embeddings

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
    # 图像嵌入生成工具函数
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate_image_embeddings_from_eeg(
        self,
        eeg_embeddings: torch.Tensor,
        n_steps: int = 100,
        solver: str = "heun",
    ) -> torch.Tensor:
        """
        使用 CFM 模型从 EEG 嵌入生成图像嵌入

        参数：
            eeg_embeddings: EEG 嵌入 [B, D]
            n_steps: CFM 采样步数
            solver: CFM 求解器类型

        返回：
            生成的图像嵌入 [B, z_dim]
        """
        if self.cfm_model is None:
            raise ValueError(
                "CFM model not loaded. Please provide cfm_checkpoint_path."
            )

        cond = self.cfm_model.retrieval(eeg_embeddings, self.eeg_encoder, top_k=3)
        z_0, z_mix = self.cfm_model.mix_prototype_prior(cond)

        # 使用 CFM 模型生成图像嵌入
        generated_embeds, _ = self.cfm_model.sample(
            cond=cond,
            z_0=z_0,
            n_steps=n_steps,
            solver=solver,
        )

        return generated_embeds

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
    def generate_images_batch(
        self,
        eeg_batch: torch.Tensor,
        *,
        top_k: int = 5,
        retain_top_n: int = 1,
        fusion_mode: FusionMode = "sd_embedding",
        num_images_per_prompt: int = 1,
        negative_prompt: str = "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        guidance_scale: Optional[float] = 12.0,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = 30,
        use_image_embeddings: bool = True,
        cfm_n_steps: int = 100,
        cfm_solver: str = "heun",
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
            use_image_embeddings: 是否使用 CFM 生成的图像嵌入。
            cfm_n_steps: CFM 采样步数。
            cfm_solver: CFM 求解器类型。

        返回：
            生成图像列表、对应的检索结果、融合后的提示词。
        """
        eeg_embeddings = self.encode_eeg_batch(eeg_batch)
        retrievals = self.retrieve_captions(eeg_embeddings, top_k=top_k)

        # 如果使用图像嵌入，生成图像嵌入
        image_embeds = None
        if use_image_embeddings and self.cfm_model is not None:
            image_embeds = self.generate_image_embeddings_from_eeg(
                eeg_embeddings, n_steps=cfm_n_steps, solver=cfm_solver
            )

        # 如果没有指定批次大小，使用整个批次
        if batch_size is None:
            batch_size = len(retrievals)

        all_images: List[Image.Image] = []
        fused_prompts: List[str] = []
        expanded_retrievals: List[List[RetrievedCaption]] = []

        # 分批处理
        for i in range(0, len(retrievals), batch_size):
            batch_retrievals = retrievals[i : i + batch_size]
            batch_start_idx = i
            batch_end_idx = min(i + batch_size, len(retrievals))

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

            effective_guidance = guidance_scale

            # 设置生成参数
            pipeline_kwargs: Dict[str, Any] = {
                "num_images_per_prompt": num_images_per_prompt,
            }
            if effective_guidance is not None:
                pipeline_kwargs["guidance_scale"] = effective_guidance
            if num_inference_steps is not None:
                pipeline_kwargs["num_inference_steps"] = num_inference_steps

            # 准备批量的 IP-Adapter 图像嵌入
            batch_ip_adapter_embeds = None
            if self.use_ip_adapter and image_embeds is not None:
                # 获取当前批次的图像嵌入
                batch_ip_adapter_embeds = image_embeds[batch_start_idx:batch_end_idx]

                # 调整嵌入形状以匹配 IP-Adapter 期望的格式
                if batch_ip_adapter_embeds.dim() == 2:
                    batch_ip_adapter_embeds = batch_ip_adapter_embeds.unsqueeze(
                        0
                    )  # [1, B, z_dim]

                neg = torch.zeros_like(batch_ip_adapter_embeds)
                batch_ip_adapter_embeds = torch.cat(
                    [neg, batch_ip_adapter_embeds], dim=0
                )

                # 确保嵌入数据类型正确
                batch_ip_adapter_embeds = batch_ip_adapter_embeds.to(self.sd_dtype)

            # 批量生成图像
            prompt_count = len(batch_fused_prompts)
            if prompt_count == 0:
                continue

            use_prompt_embeds = (
                len(batch_fused_embeds) == prompt_count and prompt_count > 0
            )

            pipeline_inputs: Dict[str, Any] = dict(pipeline_kwargs)

            if batch_ip_adapter_embeds is not None:
                pipeline_inputs["ip_adapter_image_embeds"] = [batch_ip_adapter_embeds]

            if use_prompt_embeds:
                stacked_embeds = torch.stack(batch_fused_embeds).to(
                    dtype=self.sd_pipeline.unet.dtype
                )
                pipeline_inputs["prompt_embeds"] = stacked_embeds
                if negative_prompt:
                    pipeline_inputs["negative_prompt_embeds"] = self._encode_sd_prompts(
                        [negative_prompt] * stacked_embeds.size(0)
                    )
            else:
                pipeline_inputs["prompt"] = batch_fused_prompts
                if negative_prompt:
                    pipeline_inputs["negative_prompt"] = [
                        negative_prompt
                    ] * prompt_count

            batch_images = self.sd_pipeline(**pipeline_inputs).images

            if not batch_images:
                continue

            images_per_prompt = max(1, num_images_per_prompt)
            if len(batch_images) % prompt_count != 0:
                images_per_prompt = len(batch_images) // prompt_count
                if images_per_prompt == 0:
                    images_per_prompt = 1

            for prompt_idx in range(prompt_count):
                start = prompt_idx * images_per_prompt
                end = start + images_per_prompt
                prompt_images = batch_images[start:end]
                all_images.extend(prompt_images)
                fused_prompts.extend(
                    [batch_fused_prompts[prompt_idx]] * len(prompt_images)
                )
                expanded_retrievals.extend(
                    [batch_retrievals[prompt_idx]] * len(prompt_images)
                )

        return all_images, expanded_retrievals, fused_prompts

    def compute_quality_metrics(
        self,
        generated_images: Sequence[Image.Image],
        reference_paths: Sequence[str],
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
                results["fid"] = float("nan")
        else:
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
        max_samples: int = 8,
        figsize: Optional[Tuple[int, int]] = None,
        num_generated_per_reference: int = 1,
    ) -> None:
        """
        可视化生成图像与原始图像的对比

        参数：
            generated_images: 生成的图像列表
            reference_paths: 原始图像的文件路径列表
            save_path: 保存路径，如果为None则显示图像
            max_samples: 最大可视化样本行数，默认展示8个样本
            figsize: 图像大小；若为 None，则根据行列自适应
            num_generated_per_reference: 每个原始图像对应的生成图像数
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

        if num_generated_per_reference <= 0:
            raise ValueError("num_generated_per_reference must be positive.")

        if len(generated_images) % num_generated_per_reference != 0:
            raise ValueError(
                "Generated image count must be divisible by num_generated_per_reference."
            )

        total_reference = len(generated_images) // num_generated_per_reference
        rows_to_show = min(total_reference, max_samples, 8)
        if rows_to_show == 0:
            return

        selected_indices = random.sample(range(total_reference), rows_to_show)

        cols = 1 + num_generated_per_reference
        if figsize is None:
            figsize = (cols * 2, rows_to_show * 3)

        fig, axes = plt.subplots(rows_to_show, cols, figsize=figsize)

        if rows_to_show == 1 and cols == 1:
            axes_grid = [[axes]]
        elif rows_to_show == 1:
            axes_grid = [axes]
        elif cols == 1:
            axes_grid = [[ax] for ax in axes]
        else:
            axes_grid = axes

        for row_idx, sample_idx in enumerate(selected_indices):
            start = sample_idx * num_generated_per_reference

            original_img = Image.open(reference_paths[start]).convert("RGB")
            axes_grid[row_idx][0].imshow(original_img)
            axes_grid[row_idx][0].set_title(f"Original", fontsize=10)
            axes_grid[row_idx][0].axis("off")

            for gen_offset in range(num_generated_per_reference):
                col_idx = gen_offset + 1
                img_idx = start + gen_offset
                axes_grid[row_idx][col_idx].imshow(generated_images[img_idx])
                axes_grid[row_idx][col_idx].set_title(
                    f"Generated-{gen_offset + 1}", fontsize=10
                )
                axes_grid[row_idx][col_idx].axis("off")

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
        num_validation_passes: int = 1,
    ) -> Dict[str, float]:
        """
        对验证集进行整体评估

        参数：
            val_loader: 验证集数据加载器
            max_samples: 最大评估样本数，None表示评估全部
            visualization_samples: 可视化样本数
            save_dir: 保存目录，如果为None则不保存文件
            save_interval: 每隔多少批次保存一次可视化图片，默认500
            num_validation_passes: 每个样本生成的图像数量

        返回：
            评估指标字典
        """
        import json
        import random
        from tqdm import tqdm

        if num_validation_passes <= 0:
            raise ValueError("num_validation_passes must be positive.")

        num_generations_per_sample = num_validation_passes
        # 存储所有生成的图像和对应信息
        all_generated_images = []
        all_real_images = []
        all_texts = []
        all_real_paths = []

        # 统计信息
        total_samples = len(val_loader.dataset)
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)

        expected_total = total_samples * num_generations_per_sample
        print(
            f"开始评估验证集，共计样本数: {total_samples}，"
            f"每个样本生成 {num_generations_per_sample} 张图像，预计生成总数: {expected_total}"
        )

        processed_samples = 0
        generated_image_counter = 0
        global_sample_counter = 0
        stop_processing = False
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="生成图像")):
                if stop_processing:
                    break

                eeg_batch = batch["eeg_data"]
                real_paths = batch["img_path"]

                current_batch_size = len(real_paths)
                for sample_idx in range(current_batch_size):
                    if processed_samples >= total_samples:
                        stop_processing = True
                        break

                    eeg_sample = eeg_batch[sample_idx : sample_idx + 1]
                    real_path = (
                        real_paths[sample_idx]
                        if isinstance(real_paths, (list, tuple))
                        else real_paths[sample_idx]
                    )

                    try:
                        generated_images, _, prompts = self.generate_images_batch(
                            eeg_sample,
                            top_k=5,
                            retain_top_n=3,
                            fusion_mode="sd_embedding",
                            num_images_per_prompt=num_generations_per_sample,
                            batch_size=1,
                            num_inference_steps=25,
                            use_image_embeddings=True,
                            cfm_n_steps=100,
                            guidance_scale=8,
                        )
                    except Exception as e:
                        print(
                            f"生成图像时出错 (batch {batch_idx}, sample {sample_idx}): {e}"
                        )
                        continue

                    if not generated_images:
                        continue

                    # 存储结果并保持与真实图像的对应关系
                    all_generated_images.extend(generated_images)
                    all_real_paths.extend([real_path] * len(generated_images))
                    all_texts.extend(prompts)

                    processed_samples += 1
                    global_sample_counter += 1
                    generated_image_counter += len(generated_images)

                    # 每隔 save_interval 个样本保存一次可视化图片
                    if (
                        save_dir
                        and save_interval > 0
                        and global_sample_counter % save_interval == 0
                    ):
                        try:
                            current_viz_samples = min(
                                visualization_samples, len(all_generated_images)
                            )
                            if current_viz_samples > 0:
                                save_dir_path = Path(save_dir)
                                save_dir_path.mkdir(parents=True, exist_ok=True)

                                viz_path = (
                                    save_dir_path
                                    / f"visualization_step_{global_sample_counter}.png"
                                )

                                self.visualize_samples(
                                    generated_images=all_generated_images,
                                    reference_paths=all_real_paths[
                                        : len(all_generated_images)
                                    ],
                                    save_path=viz_path,
                                    max_samples=current_viz_samples,
                                    figsize=(8, 16),
                                    num_generated_per_reference=num_generations_per_sample,
                                )
                                print(f"可视化图片已保存到: {viz_path}")
                        except Exception as e:
                            print(f"可视化保存失败 (step {global_sample_counter}): {e}")

        print(
            f"图像生成完成，共生成 {len(all_generated_images)} 张图像，"
            f"对应真实图像数量 {len(all_real_paths)}"
        )

        # 加载真实图像
        print("加载真实图像...")
        for path in tqdm(all_real_paths, desc="加载真实图像"):
            try:
                with Image.open(path) as img:
                    all_real_images.append(img.convert("RGB"))
            except Exception as e:
                print(f"加载真实图像失败 {path}: {e}")
                continue

        # 确保数量一致
        min_count = min(len(all_generated_images), len(all_real_images), len(all_texts))
        all_generated_images = all_generated_images[:min_count]
        all_real_images = all_real_images[:min_count]
        all_texts = all_texts[:min_count]

        print(f"最终评估样本数: {min_count}")

        all_texts.clear()
        all_texts = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # 计算所有指标
        print("开始计算评估指标...")
        metrics = self.compute_quality_metrics(
            generated_images=all_generated_images,
            reference_paths=all_real_paths[: len(all_generated_images)],
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

        return metrics


if __name__ == "__main__":
    import yaml
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset

    from dataset import EEGDataset, collate_fn_keep_captions
    from eeg_sd_generator import EEGStableDiffusionGenerator

    import random
    import json

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 定义需要评估的task列表
    tasks = [
        "20251028_0911_sup_with_proj_head_512_split_by_image_id_b32_sample8_prototype"
    ]

    # 准备验证集数据
    val_dataset = EEGDataset(config["dataset"], type="train")

    total_size = len(val_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    val_size = int(0.2 * total_size)
    val_indices = indices[:val_size]

    # 3. 创建子集
    val_dataset = Subset(val_dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_keep_captions,
    )

    # 对每个task进行评估
    for i, task in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"开始评估第 {i+1}/{len(tasks)} 个任务: {task}")
        print(f"{'='*50}")

        try:
            # 创建生成器
            generator = EEGStableDiffusionGenerator(
                sd_model_name="/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors",
                config=None,
                checkpoint_path=Path(
                    f"checkpoints/SupConCrossModalLoss/{task}/checkpoint_epoch_180.pt"
                ),
                text_corpus_path="/home/chengwenjie/datasets/40classes-50images/embedding/b32_caption_embeddings.pth",
                cfm_checkpoint_path="checkpoints/LowLevel/SupConCrossModalLoss/20251028_0911_sup_with_proj_head_512_split_by_image_id_b32_sample8_prototype/checkpoint_step_25000_noise_std_1.pt",
                ip_adapter_scale=0.3,
            )

            # 对验证集进行整体评估
            print(f"\n=== 验证集整体评估: {task} ===")
            val_metrics = generator.evaluate_validation_set(
                val_loader,
                max_samples=len(val_loader.dataset),
                visualization_samples=8,
                save_dir=Path(f"logs/SupConCrossModalLoss/{task}"),
                save_interval=30,
                num_validation_passes=3,
            )

            print(f"任务 {task} 验证集评估完成！")
            print("评估结果:")
            for metric_name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")

            with open(
                f"logs/SupConCrossModalLoss/{task}/metrics.json", "w", encoding="utf-8"
            ) as f:
                json.dump(val_metrics, f, indent=4)

        except Exception as e:
            print(f"评估任务 {task} 时出错: {e}")
            continue

    print(f"\n{'='*50}")
    print("所有任务评估完成！")
    print(f"{'='*50}")
