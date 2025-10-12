import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
import os
from pathlib import Path
import json
from PIL import Image
import logging

from models.eeg_encoder import EEGConformer
from models.text_encoder import CLIPTextEncoder
from retrieval.retrieval import CaptionRetriever
from weighting.weighting import CaptionWeighting, TokenFusion
from sd.pipeline import EEGToImagePipeline
from data.dataset import EEGImageDataset
from utils.metrics import calculate_all_metrics, RetrievalMetrics
from utils.seed import set_seed

logger = logging.getLogger(__name__)


class Evaluator:
    """
    模型评估器
    负责生成图像和计算各种评估指标
    """

    def __init__(self, config: dict):
        """
        初始化评估器

        Args:
            config: 配置字典
        """
        self.config = config

        # 设置随机种子
        set_seed(config["seed"])

        # 设置设备
        self.device = torch.device(config["device"])

        # 初始化组件
        self._init_models()
        self._init_retriever()
        self._init_weighting_strategies()

        # 创建输出目录
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 生成图像目录
        self.generated_dir = self.output_dir / "generated_images"
        self.generated_dir.mkdir(exist_ok=True)

    def _init_models(self):
        """初始化模型"""
        # EEG编码器
        self.eeg_encoder = EEGConformer(self.config["model"]["eeg_encoder"]).to(
            self.device
        )

        # 文本编码器
        self.text_encoder = CLIPTextEncoder(
            self.config["data"]["text"]["encoder"],
            cache_dir=self.config.get("cache_dir", None),
        )

        # Stable Diffusion管道
        self.sd_pipeline = EEGToImagePipeline(
            model_name=self.config["sd"]["model"], device=self.device
        )

        # 加载检查点
        checkpoint_path = self.config.get("checkpoint_path")
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.eeg_encoder.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("No checkpoint provided or found, using random weights")

        # 设置为评估模式
        self.eeg_encoder.eval()

    def _init_retriever(self):
        """初始化检索器"""
        self.retriever = CaptionRetriever(self.config["retrieval"])

        # 加载索引
        index_path = self.config.get("text_index_path")
        if index_path and os.path.exists(index_path):
            self.retriever.load_index(index_path)
            logger.info(f"Loaded text index from {index_path}")
        else:
            logger.warning("No text index provided or found")

    def _init_weighting_strategies(self):
        """初始化加权策略"""
        self.caption_weighting = CaptionWeighting()
        self.token_fusion = TokenFusion()

    def evaluate(self, split: str = "test") -> dict:
        """
        评估模型

        Args:
            split: 数据集分割

        Returns:
            评估结果字典
        """
        logger.info(f"Evaluating on {split} set")

        # 加载数据集
        dataset = EEGImageDataset(self.config, split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["eval"]["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # 生成图像
        generated_images, real_images, texts = self._generate_images(dataloader)

        # 计算指标
        results = calculate_all_metrics(
            generated_images=generated_images,
            real_images=real_images,
            texts=texts,
            device=self.device,
        )

        # 计算检索指标
        retrieval_results = self._evaluate_retrieval(dataloader)
        results.update(retrieval_results)

        # 保存结果
        results_path = self.output_dir / f"{split}_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f)

        logger.info(f"Results saved to {results_path}")

        return results

    def _generate_images(self, dataloader: DataLoader) -> tuple:
        """
        生成图像

        Args:
            dataloader: 数据加载器

        Returns:
            (生成图像列表, 真实图像列表, 文本列表)
        """
        generated_images = []
        real_images = []
        texts = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Generating images")
            ):
                # 获取EEG数据
                eeg_data = batch["eeg"].to(self.device)
                class_labels = batch["class_labels"]

                # EEG编码
                eeg_embeddings, class_output = self.eeg_encoder(
                    eeg_data, return_class=True
                )

                # 获取类别名称（这里简化处理）
                class_names = [f"class_{label.item()}" for label in class_labels]

                # 生成图像
                batch_images = self.sd_pipeline.generate_from_eeg(
                    eeg_embeddings=eeg_embeddings,
                    text_encoder=self.text_encoder,
                    retriever=self.retriever,
                    weighting_strategy=self.caption_weighting,
                    fusion_strategy=self.token_fusion,
                    class_names=class_names,
                    num_inference_steps=self.config["sd"]["steps"],
                    guidance_scale=self.config["sd"]["guidance_scale"],
                    height=self.config["sd"]["img_size"],
                    width=self.config["sd"]["img_size"],
                )

                # 保存生成的图像
                for i, img in enumerate(batch_images):
                    img_path = self.generated_dir / f"gen_{batch_idx}_{i}.png"
                    img.save(img_path)
                    generated_images.append(img)

                # 收集真实图像和文本
                for i, (real_img, caption_ids) in enumerate(
                    zip(batch["image"], batch["caption_ids"])
                ):
                    real_images.append(real_img)

                    # 获取第一个caption作为代表文本
                    if caption_ids:
                        # 这里简化处理，实际应该从ID获取文本
                        texts.append(f"caption_{caption_ids[0]}")
                    else:
                        texts.append("unknown")

        return generated_images, real_images, texts

    def _evaluate_retrieval(self, dataloader: DataLoader) -> dict:
        """
        评估检索性能

        Args:
            dataloader: 数据加载器

        Returns:
            检索指标字典
        """
        logger.info("Evaluating retrieval performance")

        all_eeg_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                eeg_data = batch["eeg"].to(self.device)
                class_labels = batch["class_labels"]

                # EEG编码
                eeg_embeddings, _ = self.eeg_encoder(eeg_data, return_class=False)

                all_eeg_embeddings.append(eeg_embeddings.cpu().numpy())
                all_labels.append(class_labels.numpy())

        # 合并所有embeddings
        all_eeg_embeddings = np.concatenate(all_eeg_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算检索指标
        retrieval_metrics = {
            "recall@1": RetrievalMetrics.calculate_recall_at_k(
                all_eeg_embeddings, all_eeg_embeddings, all_labels, all_labels, k=1
            ),
            "recall@5": RetrievalMetrics.calculate_recall_at_k(
                all_eeg_embeddings, all_eeg_embeddings, all_labels, all_labels, k=5
            ),
            "recall@10": RetrievalMetrics.calculate_recall_at_k(
                all_eeg_embeddings, all_eeg_embeddings, all_labels, all_labels, k=10
            ),
            "mrr": RetrievalMetrics.calculate_mrr(
                all_eeg_embeddings, all_eeg_embeddings, all_labels, all_labels
            ),
            "ndcg@10": RetrievalMetrics.calculate_ndcg(
                all_eeg_embeddings, all_eeg_embeddings, all_labels, all_labels, k=10
            ),
        }

        return retrieval_metrics

    def run_ablation_study(self) -> dict:
        """
        运行消融研究

        Returns:
            所有消融实验的结果
        """
        logger.info("Running ablation study")

        ablation_results = {}

        # 消融实验配置
        ablations = self.config.get("eval", {}).get("ablations", {})

        # 保存原始配置
        original_config = self.config.copy()

        # 句子级加权消融
        if "sentence_level" in ablations:
            sentence_results = {}
            for method in ablations["sentence_level"]:
                logger.info(f"Evaluating sentence_level={method}")

                # 更新配置
                self.config["weighting"]["sentence_level"] = method

                # 评估
                results = self.evaluate()
                sentence_results[method] = results

            ablation_results["sentence_level"] = sentence_results

        # Token级融合消融
        if "token_level" in ablations:
            token_results = {}
            for method in ablations["token_level"]:
                logger.info(f"Evaluating token_level={method}")

                # 更新配置
                self.config["weighting"]["token_level"] = method

                # 评估
                results = self.evaluate()
                token_results[method] = results

            ablation_results["token_level"] = token_results

        # 检索池消融
        if "retrieval_pool" in ablations:
            pool_results = {}
            for pool in ablations["retrieval_pool"]:
                logger.info(f"Evaluating retrieval_pool={pool}")

                # 更新配置
                self.config["data"]["retrieval_pool"] = pool

                # 评估
                results = self.evaluate()
                pool_results[pool] = results

            ablation_results["retrieval_pool"] = pool_results

        # 恢复原始配置
        self.config = original_config

        # 保存消融结果
        ablation_path = self.output_dir / "ablation_results.yaml"
        with open(ablation_path, "w") as f:
            yaml.dump(ablation_results, f)

        logger.info(f"Ablation study results saved to {ablation_path}")

        return ablation_results

    def generate_samples(self, num_samples: int = 50) -> list:
        """
        生成样本图像用于可视化

        Args:
            num_samples: 生成的样本数量

        Returns:
            生成的图像路径列表
        """
        logger.info(f"Generating {num_samples} sample images")

        # 加载数据集
        dataset = EEGImageDataset(self.config, split="test")
        dataloader = DataLoader(
            dataset, batch_size=min(num_samples, 8), shuffle=True, num_workers=4
        )

        sample_paths = []
        generated_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if generated_count >= num_samples:
                    break

                eeg_data = batch["eeg"].to(self.device)
                class_labels = batch["class_labels"]

                # EEG编码
                eeg_embeddings, class_output = self.eeg_encoder(
                    eeg_data, return_class=True
                )

                # 获取类别名称
                class_names = [f"class_{label.item()}" for label in class_labels]

                # 生成图像
                batch_images = self.sd_pipeline.generate_from_eeg(
                    eeg_embeddings=eeg_embeddings,
                    text_encoder=self.text_encoder,
                    retriever=self.retriever,
                    weighting_strategy=self.caption_weighting,
                    fusion_strategy=self.token_fusion,
                    class_names=class_names,
                    num_inference_steps=self.config["sd"]["steps"],
                    guidance_scale=self.config["sd"]["guidance_scale"],
                    height=self.config["sd"]["img_size"],
                    width=self.config["sd"]["img_size"],
                )

                # 保存图像
                for i, img in enumerate(batch_images):
                    if generated_count >= num_samples:
                        break

                    img_path = self.generated_dir / f"sample_{generated_count:04d}.png"
                    img.save(img_path)
                    sample_paths.append(str(img_path))
                    generated_count += 1

        logger.info(f"Generated {len(sample_paths)} sample images")

        return sample_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--samples", type=int, default=50)

    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["checkpoint_path"] = args.checkpoint
    config["output_dir"] = args.output_dir

    # 创建评估器
    evaluator = Evaluator(config)

    # 运行评估
    if args.ablation:
        results = evaluator.run_ablation_study()
    else:
        results = evaluator.evaluate(args.split)

        # 生成样本
        evaluator.generate_samples(args.samples)

    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")