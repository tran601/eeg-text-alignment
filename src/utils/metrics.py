import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
import torch.nn.functional as F
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3
import clip
from pytorch_fid import fid_score
import logging

logger = logging.getLogger(__name__)


class ImageMetrics:
    """
    图像生成质量评估指标
    包括FID、IS、CLIPScore等
    """

    def __init__(self, device: str = "cuda", clip_model: str = "ViT-B/32"):
        """
        初始化评估器

        Args:
            device: 计算设备
            clip_model: CLIP模型名称
        """
        self.device = device

        # 加载Inception模型用于IS和FID
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = torch.nn.Identity()  # 移除分类层
        self.inception_model.eval()
        self.inception_model.to(device)

        # 加载CLIP模型用于CLIPScore
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=device)

        # 图像预处理
        self.inception_transform = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def calculate_fid(
        self,
        generated_images: List[Image.Image],
        real_images: List[Image.Image],
        batch_size: int = 32,
    ) -> float:
        """
        计算FID分数

        Args:
            generated_images: 生成的图像列表
            real_images: 真实图像列表
            batch_size: 批处理大小

        Returns:
            FID分数（越低越好）
        """
        # 这里简化实现，实际应该使用pytorch-fid库
        # 临时保存图像到磁盘
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建子目录
            gen_dir = os.path.join(tmpdir, "generated")
            real_dir = os.path.join(tmpdir, "real")
            os.makedirs(gen_dir)
            os.makedirs(real_dir)

            # 保存图像
            for i, img in enumerate(generated_images):
                img.save(os.path.join(gen_dir, f"{i:05d}.png"))

            for i, img in enumerate(real_images):
                img.save(os.path.join(real_dir, f"{i:05d}.png"))

            # 计算FID
            fid_value = fid_score.calculate_fid_given_paths(
                [gen_dir, real_dir],
                batch_size=batch_size,
                device=self.device,
                dims=2048,
            )

        return fid_value

    def calculate_inception_score(
        self, images: List[Image.Image], batch_size: int = 32, splits: int = 10
    ) -> Tuple[float, float]:
        """
        计算Inception Score

        Args:
            images: 图像列表
            batch_size: 批处理大小
            splits: 分割数量

        Returns:
            (IS均值, IS标准差)
        """
        # 加载完整的Inception模型（包含分类层）
        inception_full = inception_v3(pretrained=True)
        inception_full.eval()
        inception_full.to(self.device)

        # 预处理图像
        processed_images = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed_images.append(self.inception_transform(img))

        # 计算预测概率
        preds = []
        with torch.no_grad():
            for i in range(0, len(processed_images), batch_size):
                batch = torch.stack(processed_images[i : i + batch_size]).to(
                    self.device
                )
                logits = inception_full(batch)
                probs = F.softmax(logits, dim=1)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # 计算Inception Score
        split_scores = []
        for i in range(splits):
            part = preds[
                i * (preds.shape[0] // splits) : (i + 1) * (preds.shape[0] // splits), :
            ]

            # 计算KL散度
            kl = part * (np.log(part) - np.log(np.expand_dims(part.mean(axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))

            split_scores.append(np.exp(kl))

        return np.mean(split_scores), np.std(split_scores)

    def calculate_clip_score(
        self, images: List[Image.Image], texts: List[str]
    ) -> float:
        """
        计算CLIPScore

        Args:
            images: 图像列表
            texts: 文本列表

        Returns:
            CLIPScore（越高越好）
        """
        if len(images) != len(texts):
            raise ValueError("Number of images and texts must match")

        # 预处理图像和文本
        image_tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_tensors.append(self.clip_preprocess(img))

        text_tokens = clip.tokenize(texts)

        # 批量处理
        batch_size = 32
        scores = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = torch.stack(image_tensors[i : i + batch_size]).to(
                    self.device
                )
                batch_texts = text_tokens[i : i + batch_size].to(self.device)

                # 获取特征
                image_features = self.clip_model.encode_image(batch_images)
                text_features = self.clip_model.encode_text(batch_texts)

                # 计算相似度
                similarity = F.cosine_similarity(image_features, text_features)
                scores.extend(similarity.cpu().numpy())

        return np.mean(scores) * 100  # 通常缩放到0-100范围

    def calculate_lpips(
        self, generated_images: List[Image.Image], real_images: List[Image.Image]
    ) -> float:
        """
        计算LPIPS（感知路径相似度）

        Args:
            generated_images: 生成的图像列表
            real_images: 真实图像列表

        Returns:
            LPIPS分数（越低越好）
        """
        try:
            import lpips
        except ImportError:
            logger.warning("LPIPS not available. Install with: pip install lpips")
            return 0.0

        # 初始化LPIPS模型
        loss_fn = lpips.LPIPS(net="alex").to(self.device)

        # 图像预处理
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # 计算距离
        distances = []
        with torch.no_grad():
            for gen_img, real_img in zip(generated_images, real_images):
                if gen_img.mode != "RGB":
                    gen_img = gen_img.convert("RGB")
                if real_img.mode != "RGB":
                    real_img = real_img.convert("RGB")

                gen_tensor = transform(gen_img).unsqueeze(0).to(self.device)
                real_tensor = transform(real_img).unsqueeze(0).to(self.device)

                distance = loss_fn(gen_tensor, real_tensor)
                distances.append(distance.item())

        return np.mean(distances)


class RetrievalMetrics:
    """
    检索性能评估指标
    包括Recall@K、MRR、NDCG等
    """

    @staticmethod
    def calculate_recall_at_k(
        query_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        query_labels: np.ndarray,
        target_labels: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        计算Recall@K

        Args:
            query_embeddings: 查询嵌入 (N, D)
            target_embeddings: 目标嵌入 (M, D)
            query_labels: 查询标签 (N,)
            target_labels: 目标标签 (M,)
            k: 返回top-k

        Returns:
            Recall@K分数
        """
        # 归一化
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        target_embeddings = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )

        # 计算相似度矩阵
        sim_matrix = np.dot(query_embeddings, target_embeddings.T)

        # 计算Recall@K
        correct = 0
        for i in range(len(query_embeddings)):
            # 获取top-k最相似的目标
            top_k_idx = np.argpartition(sim_matrix[i], -k)[-k:]
            top_k_labels = target_labels[top_k_idx]

            # 检查是否有同类标签
            if query_labels[i] in top_k_labels:
                correct += 1

        return correct / len(query_embeddings)

    @staticmethod
    def calculate_mrr(
        query_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        query_labels: np.ndarray,
        target_labels: np.ndarray,
    ) -> float:
        """
        计算平均倒数排名(MRR)

        Args:
            query_embeddings: 查询嵌入 (N, D)
            target_embeddings: 目标嵌入 (M, D)
            query_labels: 查询标签 (N,)
            target_labels: 目标标签 (M,)

        Returns:
            MRR分数
        """
        # 归一化
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        target_embeddings = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )

        # 计算相似度矩阵
        sim_matrix = np.dot(query_embeddings, target_embeddings.T)

        # 计算MRR
        reciprocal_ranks = []
        for i in range(len(query_embeddings)):
            # 按相似度排序
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            sorted_labels = target_labels[sorted_idx]

            # 找到第一个匹配的位置
            for rank, label in enumerate(sorted_labels, 1):
                if label == query_labels[i]:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    @staticmethod
    def calculate_ndcg(
        query_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        query_labels: np.ndarray,
        target_labels: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        计算NDCG@K

        Args:
            query_embeddings: 查询嵌入 (N, D)
            target_embeddings: 目标嵌入 (M, D)
            query_labels: 查询标签 (N,)
            target_labels: 目标标签 (M,)
            k: 返回top-k

        Returns:
            NDCG@K分数
        """
        # 归一化
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        target_embeddings = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )

        # 计算相似度矩阵
        sim_matrix = np.dot(query_embeddings, target_embeddings.T)

        # 计算NDCG@K
        ndcg_scores = []
        for i in range(len(query_embeddings)):
            # 按相似度排序
            sorted_idx = np.argsort(sim_matrix[i])[::-1][:k]
            sorted_labels = target_labels[sorted_idx]

            # 计算DCG
            dcg = 0.0
            for rank, label in enumerate(sorted_labels, 1):
                if label == query_labels[i]:
                    dcg += 1.0 / np.log2(rank + 1)
                    break

            # 计算IDCG（理想DCG，假设第一个结果就是相关的）
            idcg = 1.0  # 因为只有第一个位置是相关的

            # 计算NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores)


def calculate_all_metrics(
    generated_images: List[Image.Image],
    real_images: List[Image.Image],
    texts: List[str],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        generated_images: 生成的图像列表
        real_images: 真实图像列表
        texts: 对应的文本列表
        device: 计算设备

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # 初始化评估器
    evaluator = ImageMetrics(device=device)

    # 计算FID
    try:
        fid = evaluator.calculate_fid(generated_images, real_images)
        metrics["FID"] = fid
        logger.info(f"FID: {fid:.4f}")
    except Exception as e:
        logger.warning(f"Failed to calculate FID: {e}")

    # 计算IS
    try:
        is_mean, is_std = evaluator.calculate_inception_score(generated_images)
        metrics["IS_mean"] = is_mean
        metrics["IS_std"] = is_std
        logger.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        logger.warning(f"Failed to calculate Inception Score: {e}")

    # 计算CLIPScore
    try:
        clip_score = evaluator.calculate_clip_score(generated_images, texts)
        metrics["CLIPScore"] = clip_score
        logger.info(f"CLIPScore: {clip_score:.4f}")
    except Exception as e:
        logger.warning(f"Failed to calculate CLIPScore: {e}")

    # 计算LPIPS
    try:
        lpips = evaluator.calculate_lpips(generated_images, real_images)
        metrics["LPIPS"] = lpips
        logger.info(f"LPIPS: {lpips:.4f}")
    except Exception as e:
        logger.warning(f"Failed to calculate LPIPS: {e}")

    return metrics