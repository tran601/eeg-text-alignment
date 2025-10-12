import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Dict, Any
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class EEGToImagePipeline:
    """
    EEG到图像生成的管道
    将EEG编码的文本条件输入到Stable Diffusion中生成图像
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        初始化管道

        Args:
            model_name: SD模型名称
            device: 设备
            torch_dtype: 数据类型
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # 加载Stable Diffusion管道
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,  # 禁用安全检查器以获得更多控制
            requires_safety_checker=False,
        ).to(device)

        # 设置为评估模式（StableDiffusionPipeline没有eval方法，但可以设置其组件）
        if hasattr(self.pipe, "unet") and hasattr(self.pipe.unet, "eval"):
            self.pipe.unet.eval()
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "eval"):
            self.pipe.vae.eval()
        if hasattr(self.pipe, "text_encoder") and hasattr(
            self.pipe.text_encoder, "eval"
        ):
            self.pipe.text_encoder.eval()

        # 默认生成参数
        self.default_params = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 256,
            "width": 256,
            "eta": 0.0,
            "generator": None,
        }

    def generate_from_embeddings(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ) -> List[Image.Image]:
        """
        从文本嵌入生成图像

        Args:
            prompt_embeds: 正向提示嵌入 (batch_size, 77, 768)
            negative_prompt_embeds: 负向提示嵌入 (batch_size, 77, 768)
            num_images_per_prompt: 每个提示生成的图像数量
            **kwargs: 其他生成参数

        Returns:
            生成的图像列表
        """
        # 合并参数
        params = {**self.default_params, **kwargs}

        # 确保嵌入在正确的设备和数据类型上
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.torch_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=self.device, dtype=self.torch_dtype
            )

        # 批量生成
        batch_size = prompt_embeds.shape[0]
        images = []

        with torch.no_grad():
            for i in range(batch_size):
                # 获取当前样本的嵌入
                current_prompt_embeds = prompt_embeds[i : i + 1]
                current_negative_embeds = None

                if negative_prompt_embeds is not None:
                    current_negative_embeds = negative_prompt_embeds[i : i + 1]

                # 生成图像
                batch_images = self.pipe(
                    prompt_embeds=current_prompt_embeds,
                    negative_prompt_embeds=current_negative_embeds,
                    num_images_per_prompt=num_images_per_prompt,
                    **params,
                ).images

                images.extend(batch_images)

        return images

    def generate_from_eeg(
        self,
        eeg_embeddings: torch.Tensor,
        text_encoder,
        retriever,
        weighting_strategy,
        fusion_strategy,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        从EEG嵌入生成图像（完整流程）

        Args:
            eeg_embeddings: EEG嵌入 (batch_size, 768)
            text_encoder: 文本编码器
            retriever: 检索器
            weighting_strategy: 加权策略
            fusion_strategy: 融合策略
            class_names: 类别名称列表
            **kwargs: 其他生成参数

        Returns:
            生成的图像列表
        """
        batch_size = eeg_embeddings.shape[0]
        prompt_embeds_list = []
        negative_prompt_embeds_list = []

        # 对每个EEG样本进行处理
        for i in range(batch_size):
            eeg_embed = eeg_embeddings[i].cpu().numpy()

            # 1. 检索相关captions
            hits = retriever.search_topk(eeg_embed, k=10)

            # 2. 如果有类别信息，进行重排
            if class_names is not None:
                # 获取类别嵌入
                class_name = class_names[i] if i < len(class_names) else "object"
                class_embed = text_encoder.encode_sentence(class_name)

                # 重排
                hits = retriever.rerank_by_class(
                    hits, eeg_embed, class_embed, gamma=0.3, top_r=3
                )

            # 3. 获取候选captions的token嵌入
            caption_ids = [hit.caption_id for hit in hits]
            token_matrices = []

            for caption_id in caption_ids:
                # 这里简化处理，实际应该从caption_id获取文本
                # 假设我们有从ID到文本的映射
                caption_text = f"caption_{caption_id}"  # 占位符
                token_embed = text_encoder.encode_tokens(caption_text)
                token_matrices.append(token_embed)

            # 4. 句子级加权
            candidate_embeddings = np.stack([hit.embedding for hit in hits])
            sentence_weights = weighting_strategy.sparse_convex_combination(
                eeg_embed, candidate_embeddings, method="nnls"
            )

            # 5. Token级融合
            if hasattr(fusion_strategy, "base_align_fusion"):
                fused_tokens = fusion_strategy.base_align_fusion(
                    token_matrices, sentence_weights, eeg_embed
                )
            else:
                # 默认使用广播融合
                fused_tokens = fusion_strategy.broadcast_fusion(
                    token_matrices, sentence_weights
                )

            # 6. 类别向量融合（SLERP）
            if class_names is not None:
                class_prompt_embed = text_encoder.encode_class_prompt(class_name)
                fused_tokens = self.slerp_tokens(
                    class_prompt_embed, fused_tokens, t=0.5
                )

            # 转换为torch tensor
            prompt_embed = torch.from_numpy(fused_tokens).unsqueeze(0)
            prompt_embeds_list.append(prompt_embed)

        # 批量生成图像
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)

        # 生成负向提示（可以简单使用空提示）
        negative_prompt = ""
        with torch.no_grad():
            negative_inputs = text_encoder.tokenizer(
                [negative_prompt] * batch_size,
                padding="max_length",
                truncation=True,
                max_length=text_encoder.max_length,
                return_tensors="pt",
            )

            negative_outputs = text_encoder.text_model(**negative_inputs)
            negative_prompt_embeds = negative_outputs.last_hidden_state.to(
                device=self.device, dtype=self.torch_dtype
            )

        # 生成图像
        images = self.generate_from_embeddings(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            **kwargs,
        )

        return images

    def slerp_tokens(
        self, token_a: np.ndarray, token_b: np.ndarray, t: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        球面线性插值(SLERP)两个token矩阵

        Args:
            token_a: 第一个token矩阵 (77, 768)
            token_b: 第二个token矩阵 (77, 768)
            t: 插值系数，可以是标量或向量

        Returns:
            插值后的token矩阵 (77, 768)
        """
        # 归一化
        token_a_norm = token_a / np.linalg.norm(token_a, axis=1, keepdims=True)
        token_b_norm = token_b / np.linalg.norm(token_b, axis=1, keepdims=True)

        # 计算点积（余弦相似度）
        dot_product = np.sum(token_a_norm * token_b_norm, axis=1, keepdims=True)

        # 限制在[-1, 1]范围内以避免数值误差
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # 计算角度
        theta = np.arccos(dot_product)

        # 如果角度很小，使用线性插值
        small_angle = theta < 1e-6
        result = np.zeros_like(token_a)

        # 小角度情况
        if np.any(small_angle):
            t_broadcast = t if isinstance(t, np.ndarray) else t
            result[small_angle.squeeze()] = (1 - t_broadcast) * token_a[
                small_angle.squeeze()
            ] + t_broadcast * token_b[small_angle.squeeze()]

        # 一般情况
        if not np.all(small_angle):
            # 非小角度的索引
            normal_idx = ~small_angle.squeeze()

            # 获取对应的t值
            if isinstance(t, np.ndarray):
                t_normal = t[normal_idx]
            else:
                t_normal = t

            # SLERP公式
            sin_theta = np.sin(theta[normal_idx])
            a_weights = np.sin((1 - t_normal) * theta[normal_idx]) / sin_theta
            b_weights = np.sin(t_normal * theta[normal_idx]) / sin_theta

            result[normal_idx] = (
                a_weights * token_a[normal_idx] + b_weights * token_b[normal_idx]
            )

        return result

    def enable_xformers_memory_efficient_attention(self):
        """启用xformers内存高效注意力"""
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        else:
            logger.warning("xformers memory efficient attention not available")

    def enable_cpu_offload(self):
        """启用CPU卸载"""
        if hasattr(self.pipe, "enable_sequential_cpu_offload"):
            self.pipe.enable_sequential_cpu_offload()
            logger.info("Enabled CPU offload")
        else:
            logger.warning("CPU offload not available")

    def enable_model_cpu_offload(self):
        """启用模型CPU卸载"""
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        else:
            logger.warning("Model CPU offload not available")