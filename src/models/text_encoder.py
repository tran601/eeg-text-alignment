import torch
import numpy as np
import time
import os
from typing import List, Optional
from diffusers import StableDiffusionPipeline


class CLIPTextEncoder:
    """
    CLIP文本编码器
    使用Stable Diffusion管道内部的tokenizer和text_encoder进行文本编码
    """

    def __init__(
        self,
        model_name: str = "/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        初始化CLIP文本编码器

        Args:
            model_name: SD模型目录
            torch_dtype: 数据类型
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype

        # 初始化Stable Diffusion管道以获取内部的tokenizer和text_encoder
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        )

        # 获取tokenizer和text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)

        # 最大长度
        self.max_length = self.tokenizer.model_max_length

        # 缓存
        self.sentence_cache = {}
        self.token_cache = {}

    def encode_sentence(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        编码单个句子，返回句子级嵌入

        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            句子嵌入 (768,)
        """
        if use_cache and text in self.sentence_cache:
            return self.sentence_cache[text]

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

            # 获取输出
            outputs = self.text_encoder(inputs.input_ids.to(self.device))

            # 使用pooled output作为句子表示
            sentence_embed = outputs.pooler_output.squeeze(0).cpu().numpy()

            # 缓存
            if use_cache:
                self.sentence_cache[text] = sentence_embed

            return sentence_embed

    def encode_tokens(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        编码单个句子，返回token级嵌入

        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            token嵌入 (77, 768)
        """
        if use_cache and text in self.token_cache:
            return self.token_cache[text]

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

            # 获取输出 - 使用[0]获取last_hidden_state
            token_embeds = self.text_encoder(inputs.input_ids.to(self.device))[0]
            token_embeds = token_embeds.squeeze(0).cpu().numpy()

            # 缓存
            if use_cache:
                self.token_cache[text] = token_embeds

            return token_embeds

    def encode_batch_sentences(self, texts: List[str]) -> np.ndarray:
        """
        批量编码句子

        Args:
            texts: 文本列表

        Returns:
            句子嵌入 (N, 768)
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 获取输出
            outputs = self.text_encoder(inputs.input_ids.to(self.device))

            # 使用pooled output
            sentence_embeds = outputs.pooler_output.cpu().numpy()

            # L2归一化
            sentence_embeds = sentence_embeds / np.linalg.norm(
                sentence_embeds, axis=1, keepdims=True
            )

            return sentence_embeds

    def encode_batch_tokens(self, texts: List[str]) -> np.ndarray:
        """
        批量编码token

        Args:
            texts: 文本列表

        Returns:
            token嵌入 (N, 77, 768)
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 获取输出 - 使用[0]获取last_hidden_state
            token_embeds = self.text_encoder(inputs.input_ids.to(self.device))[0]
            token_embeds = token_embeds.cpu().numpy()

            # 不进行L2归一化，保持与SD管道一致
            # token_embeds = token_embeds / np.linalg.norm(
            #     token_embeds, axis=2, keepdims=True
            # )

            return token_embeds

    def encode_class_prompt(
        self, class_name: str, template: str = "a photo of {class_name}"
    ) -> np.ndarray:
        """
        编码类别提示

        Args:
            class_name: 类别名称
            template: 提示模板

        Returns:
            token嵌入 (77, 768)
        """
        prompt = template.format(class_name=class_name)
        return self.encode_tokens(prompt)

    def clear_cache(self):
        """清空缓存"""
        self.sentence_cache.clear()
        self.token_cache.clear()


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.sd.pipeline import EEGToImagePipeline
    from PIL import Image
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CLIPTextEncoder 测试")
    print("=" * 60)

    # 初始化文本编码器
    print("\n1. 初始化文本编码器...")
    text_encoder = CLIPTextEncoder()
    print(f"设备: {text_encoder.device}")
    print(f"数据类型: {text_encoder.torch_dtype}")
    print(f"最大长度: {text_encoder.max_length}")

    # 测试文本
    test_texts = [
        "a beautiful landscape with mountains",
        "a cute cat sitting on a table",
        "a futuristic cityscape at night",
        "a beautiful landscape with mountains",  # 重复文本，测试缓存
    ]

    # 测试句子级编码
    print("\n2. 测试句子级编码...")
    for i, text in enumerate(test_texts):
        start_time = time.time()
        sentence_embed = text_encoder.encode_sentence(text)
        end_time = time.time()

        print(f"文本 {i+1}: '{text}'")
        print(f"  嵌入形状: {sentence_embed.shape}")
        print(f"  嵌入范数: {np.linalg.norm(sentence_embed):.4f}")
        print(f"  处理时间: {end_time - start_time:.4f}秒")
        print(f"  {'(使用缓存)' if i == 3 else '(首次计算)'}")

    # 测试token级编码
    print("\n3. 测试token级编码...")
    for i, text in enumerate(test_texts[:3]):  # 只测试前3个
        start_time = time.time()
        token_embed = text_encoder.encode_tokens(text)
        end_time = time.time()

        print(f"文本 {i+1}: '{text}'")
        print(f"  嵌入形状: {token_embed.shape}")
        print(f"  处理时间: {end_time - start_time:.4f}秒")

    # 测试类别提示编码
    print("\n4. 测试类别提示编码...")
    class_names = ["cat", "dog", "car", "airplane"]
    for class_name in class_names:
        class_embed = text_encoder.encode_class_prompt(class_name)
        print(f"类别: {class_name}, 嵌入形状: {class_embed.shape}")

    # 测试批量编码
    print("\n5. 测试批量编码...")
    batch_texts = test_texts[:3]
    start_time = time.time()
    batch_sentence_embeds = text_encoder.encode_batch_sentences(batch_texts)
    batch_token_embeds = text_encoder.encode_batch_tokens(batch_texts)
    end_time = time.time()

    print(f"批量句子编码形状: {batch_sentence_embeds.shape}")
    print(f"批量token编码形状: {batch_token_embeds.shape}")
    print(f"批量处理时间: {end_time - start_time:.4f}秒")

    # 测试使用token级嵌入生成图像
    print("\n6. 测试使用token级嵌入生成图像...")
    try:
        # 初始化SD管道
        # 测试提示词
        test_prompt = "a beautiful landscape with mountains and a lake"

        # 获取token级嵌入
        print(f"编码提示词: '{test_prompt}'")
        token_embeds = text_encoder.encode_tokens(test_prompt)
        token_embeds_tensor = (
            torch.from_numpy(token_embeds)
            .unsqueeze(0)
            .to(device=text_encoder.device, dtype=text_encoder.torch_dtype)
        )

        print(f"Token嵌入形状: {token_embeds_tensor.shape}")

        # 生成图像
        print("生成图像...")
        start_time = time.time()
        text_encoder.pipe = text_encoder.pipe.to(text_encoder.device)
        images = text_encoder.pipe(
            prompt_embeds=token_embeds_tensor,
            num_inference_steps=50,  # 减少步数以加快测试
            guidance_scale=7.5,
        )[0]
        end_time = time.time()

        print(f"图像生成时间: {end_time - start_time:.2f}秒")
        print(f"生成的图像数量: {len(images)}")

        # 保存图像
        output_dir = project_root / "test_outputs"
        output_dir.mkdir(exist_ok=True)

        for i, img in enumerate(images):
            output_path = output_dir / f"text_encoder_test_{i}.png"
            img.save(output_path)
            print(f"图像已保存到: {output_path}")

        # 显示图像信息
        if images:
            img = images[0]
            print(f"图像尺寸: {img.size}")
            print(f"图像模式: {img.mode}")

    except Exception as e:
        print(f"图像生成失败: {str(e)}")
        print("可能的原因: SD模型路径不正确或GPU内存不足")

    # 测试缓存效果
    print("\n7. 测试缓存效果...")
    test_text = "a beautiful landscape with mountains"

    # 清空缓存
    text_encoder.clear_cache()
    print("缓存已清空")

    # 首次编码（无缓存）
    start_time = time.time()
    text_encoder.encode_sentence(test_text)
    first_time = time.time() - start_time
    print(f"首次编码时间: {first_time:.4f}秒")

    # 二次编码（使用缓存）
    start_time = time.time()
    text_encoder.encode_sentence(test_text)
    cached_time = time.time() - start_time
    print(f"缓存编码时间: {cached_time:.4f}秒")
    print(f"速度提升: {first_time/cached_time:.2f}x")

    # 显示缓存状态
    print(f"\n缓存状态:")
    print(f"句子缓存条目数: {len(text_encoder.sentence_cache)}")
    print(f"Token缓存条目数: {len(text_encoder.token_cache)}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)