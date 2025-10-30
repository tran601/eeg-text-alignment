"""
Dual-encoder EEG-driven Stable Diffusion generator.

This module reproduces the capabilities of ``eeg_sd_generator.py`` while
replacing the image embedding pathway with a second EEG encoder trained by
``train_eeg_alignment.py``. The retrieval encoder continues to fetch and fuse
captions, whereas the alignment encoder outputs IP-Adapter embeddings directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numbers
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models import CLIPTextEncoder, EEGEncoder
from utils.metrics import ImageMetrics

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "diffusers is required for Stable Diffusion generation. "
        "Install it with `pip install diffusers`."
    ) from exc


FusionMode = Literal["text", "sd_embedding"]


@dataclass
class RetrievedCaption:
    caption: str
    similarity: float
    image_id: int
    corpus_index: int


class EEGDualEncoderStableDiffusionGenerator:
    """
    Stable Diffusion generator using two EEG encoders:

    - ``self.eeg_encoder`` handles caption retrieval and fusion (same as the
      original generator);
    - ``self.alignment_eeg_encoder`` converts EEG data to IP-Adapter embeddings
      without relying on the CFM module.
    """

    def __init__(
        self,
        *,
        alignment_checkpoint_path: Path,
        checkpoint_path: Path,
        config: Optional[Dict[str, Any]] = None,
        text_corpus_path: Optional[Path] = None,
        sd_model_name: str = "/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5",
        device: Optional[str] = None,
        sd_dtype: torch.dtype = torch.float16,
        enable_xformers: bool = False,
        use_ip_adapter: bool = True,
        ip_adapter_scale: float = 0.2,
    ) -> None:
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

        alignment_checkpoint = torch.load(alignment_checkpoint_path, map_location="cpu")
        alignment_cfg = alignment_checkpoint["config"]["model"]["eeg_encoder"]
        self.alignment_eeg_encoder = EEGEncoder(alignment_cfg).to(self.device)
        self.alignment_eeg_encoder.load_state_dict(
            alignment_checkpoint["eeg_encoder_state_dict"]
        )
        self.alignment_eeg_encoder.eval()

        text_cfg = config["model"]["text_encoder"]
        self.clip_text_encoder = CLIPTextEncoder(text_cfg["model_path"])

        corpus_path = Path(
            text_corpus_path
            or config.get("retrieval", {}).get("full_caption_corpus_path", "")
        )
        if not corpus_path:
            raise ValueError("caption corpus path must be provided.")
        self.text_corpus = self._build_text_corpus(corpus_path)

        with torch.no_grad():
            projected = self.eeg_encoder.text_projector(
                self.text_corpus["text_vectors"].to(self.device)
            )
            self.projected_text_vectors = F.normalize(projected, dim=-1)

        self.sd_pipeline = StableDiffusionPipeline.from_single_file(
            sd_model_name, torch.dtype=torch.float16, variant="fp16"
        ).to(self.device)

        self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipeline.scheduler.config, use_karras_sigmas=True
        )

        if enable_xformers and hasattr(
            self.sd_pipeline, "enable_xformers_memory_efficient_attention"
        ):
            self.sd_pipeline.enable_xformers_memory_efficient_attention()

        self.use_ip_adapter = use_ip_adapter
        self.ip_adapter_scale = ip_adapter_scale
        if self.use_ip_adapter:
            self.sd_pipeline.load_ip_adapter(
                "/home/chengwenjie/workspace/models/ip-adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            self.sd_pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

        self.image_metrics = ImageMetrics(device=str(self.device))

    def _load_checkpoint(
        self, checkpoint_data: Dict[str, Any], checkpoint_path: Path
    ) -> None:
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
        if not path.exists():
            raise FileNotFoundError(f"Caption corpus not found: {path}")
        corpus = torch.load(path, map_location="cpu")
        return {
            "image_id": corpus["image_ids"].long(),
            "caption": corpus["captions"],
            "text_vectors": corpus["embeddings"].float(),
            "image_paths": corpus.get("image_paths", []),
        }

    @torch.no_grad()
    def encode_eeg_batch(self, eeg_batch: torch.Tensor) -> torch.Tensor:
        eeg_batch = eeg_batch.to(self.device)
        embeds, _ = self.eeg_encoder(eeg_batch, return_class=False)
        return F.normalize(embeds, dim=-1)

    @torch.no_grad()
    def retrieve_captions(
        self,
        eeg_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> List[List[RetrievedCaption]]:
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

    @torch.no_grad()
    def generate_image_embeddings_from_eeg(
        self,
        eeg_batch: torch.Tensor,
    ) -> torch.Tensor:
        eeg_batch = eeg_batch.to(self.device)
        embeds, _ = self.alignment_eeg_encoder(eeg_batch, return_class=False)
        return embeds

    def _softmax_weights(self, similarities: Sequence[float]) -> torch.Tensor:
        sims = torch.tensor(similarities, dtype=torch.float32, device=self.device)
        return torch.softmax(sims, dim=0)

    def _encode_sd_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
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
        if retain_top_n <= 0:
            raise ValueError("retain_top_n must be positive.")
        if retain_top_n > len(captions):
            raise ValueError("retain_top_n exceeds available captions.")

        selected = captions[:retain_top_n]
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
    ) -> Tuple[List[Image.Image], List[List[RetrievedCaption]], List[str]]:
        eeg_embeddings = self.encode_eeg_batch(eeg_batch)
        retrievals = self.retrieve_captions(eeg_embeddings, top_k=top_k)

        image_embeds = None
        if use_image_embeddings:
            image_embeds = self.generate_image_embeddings_from_eeg(eeg_batch)

        if batch_size is None:
            batch_size = len(retrievals)

        all_images: List[Image.Image] = []
        fused_prompts: List[str] = []
        expanded_retrievals: List[List[RetrievedCaption]] = []

        for i in range(0, len(retrievals), batch_size):
            batch_retrievals = retrievals[i : i + batch_size]
            batch_start_idx = i
            batch_end_idx = min(i + batch_size, len(retrievals))

            batch_fused_prompts: List[str] = []
            batch_fused_embeds: List[torch.Tensor] = []

            for captions in batch_retrievals:
                fused_prompt, fused_embed, _ = self.fuse_captions(
                    captions, retain_top_n=retain_top_n, mode=fusion_mode
                )
                batch_fused_prompts.append(fused_prompt)
                if fused_embed is not None:
                    batch_fused_embeds.append(fused_embed.squeeze(0))

            pipeline_kwargs: Dict[str, Any] = {
                "num_images_per_prompt": num_images_per_prompt,
            }
            if guidance_scale is not None:
                pipeline_kwargs["guidance_scale"] = guidance_scale
            if num_inference_steps is not None:
                pipeline_kwargs["num_inference_steps"] = num_inference_steps

            batch_ip_adapter_embeds = None
            if self.use_ip_adapter and image_embeds is not None:
                batch_ip_adapter_embeds = image_embeds[batch_start_idx:batch_end_idx]
                if batch_ip_adapter_embeds.dim() == 2:
                    batch_ip_adapter_embeds = batch_ip_adapter_embeds.unsqueeze(0)

                neg = torch.zeros_like(batch_ip_adapter_embeds)
                batch_ip_adapter_embeds = torch.cat(
                    [neg, batch_ip_adapter_embeds], dim=0
                )
                batch_ip_adapter_embeds = batch_ip_adapter_embeds.to(self.sd_dtype)

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
            except ValueError:
                results["fid"] = float("nan")
        else:
            results["fid"] = float("nan")

        is_mean, is_std = self.image_metrics.calculate_inception_score(generated_images)
        results["is_mean"] = is_mean
        results["is_std"] = is_std

        results["clip_score"] = self.image_metrics.calculate_clip_score(
            generated_images, real_images
        )
        results["ssim"] = self.image_metrics.calculate_ssim(
            generated_images, real_images
        )
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
        import matplotlib.pyplot as plt
        import matplotlib
        import random

        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "DejaVu Sans",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

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
            axes_grid[row_idx][0].set_title("Original", fontsize=10)
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

        plt.close()

    def evaluate_validation_set(
        self,
        val_loader,
        *,
        max_samples: Optional[int] = None,
        visualization_samples: int = 16,
        save_dir: Optional[Path] = None,
        save_interval: int = 50,
        num_validation_passes: int = 1,
        num_generations_per_sample: int = 1,
    ) -> Dict[str, float]:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        if visualization_samples <= 0:
            raise ValueError("visualization_samples must be positive.")
        if num_validation_passes <= 0:
            raise ValueError("num_validation_passes must be positive.")
        if num_generations_per_sample <= 0:
            raise ValueError("num_generations_per_sample must be positive.")

        all_generated_images: List[Image.Image] = []
        all_real_paths: List[str] = []
        all_real_images: List[Image.Image] = []
        all_texts: List[str] = []

        processed_samples = 0
        total_samples = len(val_loader.dataset)
        sample_limit = max_samples or total_samples

        print(f"开始评估，共进行 {num_validation_passes} 次遍历。")

        global_sample_counter = 0
        generated_image_counter = 0

        for pass_idx in range(num_validation_passes):
            print(f"=== 第 {pass_idx + 1}/{num_validation_passes} 次遍历 ===")

            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc=f"Pass {pass_idx + 1}", leave=False)
            ):
                eeg_data = batch["eeg_data"]
                real_paths = batch["img_path"]

                for sample_idx in range(len(eeg_data)):
                    if processed_samples >= sample_limit:
                        break

                    eeg_sample = eeg_data[sample_idx].unsqueeze(0)
                    real_path = (
                        real_paths[sample_idx]
                        if isinstance(real_paths[sample_idx], str)
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
                            guidance_scale=8,
                        )
                    except Exception as e:
                        print(
                            f"生成图像时出错 (batch {batch_idx}, sample {sample_idx}): {e}"
                        )
                        continue

                    if not generated_images:
                        continue

                    all_generated_images.extend(generated_images)
                    all_real_paths.extend([real_path] * len(generated_images))
                    all_texts.extend(prompts)

                    processed_samples += 1
                    global_sample_counter += 1
                    generated_image_counter += len(generated_images)

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

                if processed_samples >= sample_limit:
                    break

        print(
            f"图像生成完成，共生成 {len(all_generated_images)} 张图像，"
            f"对应真实图像数量 {len(all_real_paths)}"
        )

        print("加载真实图像...")
        for path in tqdm(all_real_paths, desc="加载真实图像"):
            try:
                with Image.open(path) as img:
                    all_real_images.append(img.convert("RGB"))
            except Exception as e:
                print(f"加载真实图像失败 {path}: {e}")
                continue

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

        print("开始计算评估指标...")
        metrics = self.compute_quality_metrics(
            generated_images=all_generated_images,
            reference_paths=all_real_paths[: len(all_generated_images)],
        )
        metrics = {
            key: float(value) if isinstance(value, numbers.Number) else value
            for key, value in metrics.items()
        }

        print("=== 验证集评估结果 ===")
        for metric_name, value in metrics.items():
            if isinstance(value, numbers.Number):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")

        return metrics


def _save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import random
    import yaml
    from torch.utils.data import DataLoader, Subset

    from dataset import EEGDataset, collate_fn_keep_captions

    config_path = Path("configs/default.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    evaluation_tasks = [
        {
            "name": "20251028_0911_sup_with_proj_head_512_split_by_image_id_b32_sample8_prototype",
            "retrieval_checkpoint": Path(
                "checkpoints/SupConCrossModalLoss/20251028_0911_sup_with_proj_head_512_split_by_image_id_b32_sample8_prototype/checkpoint_epoch_180.pt"
            ),
            "alignment_checkpoint": Path(
                "checkpoints/EEGAlignment/MSECosine/20251030_085206/checkpoint_step_15000.pt"
            ),
            "sd_model": Path(
                "/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
            ),
            "ip_adapter_scale": 0.3,
        }
    ]

    dataset = EEGDataset(base_config["dataset"], type="train")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_size = int(0.2 * len(indices))
    subset = Subset(dataset, indices[:subset_size])

    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_keep_captions,
    )

    for task in evaluation_tasks:
        print("\n" + "=" * 50)
        print(f"Evaluating task: {task['name']}")
        print("=" * 50)

        generator = EEGDualEncoderStableDiffusionGenerator(
            alignment_checkpoint_path=task["alignment_checkpoint"],
            checkpoint_path=task["retrieval_checkpoint"],
            config=None,
            text_corpus_path=Path(base_config["retrieval"]["full_caption_corpus_path"]),
            sd_model_name=str(task["sd_model"]),
            ip_adapter_scale=task.get("ip_adapter_scale", 0.2),
        )

        output_dir = Path("logs/DualEncoder") / task["name"]
        metrics = generator.evaluate_validation_set(
            loader,
            max_samples=len(subset),
            visualization_samples=8,
            save_dir=output_dir,
            save_interval=30,
            num_validation_passes=3,
        )

        _save_metrics(metrics, output_dir / "metrics.json")
        print("Metrics saved to", output_dir / "metrics.json")

        del generator
        torch.cuda.empty_cache()
