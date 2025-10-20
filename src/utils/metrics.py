import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
import torch.nn.functional as F
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import logging

from transformers import CLIPModel, CLIPImageProcessor

# 添加SSIM相关导入
from pytorch_msssim import ssim

logger = logging.getLogger(__name__)


class ImageMetrics:
    """
    图像生成质量评估指标
    包括FID、IS、CLIPScore等
    """

    def __init__(
        self,
        device: str = "cuda",
        clip_model: str = "/home/chengwenjie/workspace/models/CLIP-ViT-B-32-laion2B-s34B-b79K",
    ):
        """
        初始化评估器

        Args:
            device: 计算设备
            clip_model: Hugging Face CLIP模型名称或本地路径
        """
        self.device = device

        # 加载Inception模型用于IS和FID
        inception_weights = Inception_V3_Weights.DEFAULT
        self.inception_model = inception_v3(
            weights=inception_weights, transform_input=False
        )
        self.inception_model.fc = torch.nn.Identity()  # 移除分类层
        self.inception_model.eval()
        self.inception_model.to(device)
        self._inception_classifier: Optional[torch.nn.Module] = None
        self._inception_weights = inception_weights

        # 加载CLIP模型与预处理器用于CLIPScore
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(device)
        self.clip_model.eval()
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model)

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
        if len(generated_images) < 2 or len(real_images) < 2:
            raise ValueError("FID requires at least two generated and real images.")

        gen_features = self._compute_inception_activations(generated_images, batch_size)
        real_features = self._compute_inception_activations(real_images, batch_size)
        if gen_features.ndim != 2 or real_features.ndim != 2:
            raise ValueError(
                f"Features must be 2D. Got {gen_features.shape=} {real_features.shape=}"
            )
        if gen_features.shape[1] != real_features.shape[1]:
            raise ValueError(
                f"Feature dims must match. Got {gen_features.shape[1]} vs {real_features.shape[1]}"
            )

        if not np.isfinite(gen_features).all():
            raise ValueError("gen_features contains NaN/Inf.")
        if not np.isfinite(real_features).all():
            raise ValueError("real_features contains NaN/Inf.")

        # 用 float64 计算均值和协方差，稳定性更好
        gen_features64 = gen_features.astype(np.float64, copy=False)
        real_features64 = real_features.astype(np.float64, copy=False)

        mu_gen = np.mean(gen_features64, axis=0)
        mu_real = np.mean(real_features64, axis=0)

        sigma_gen = np.cov(gen_features64, rowvar=False)
        sigma_real = np.cov(real_features64, rowvar=False)

        # 检查
        if not (np.isfinite(mu_gen).all() and np.isfinite(mu_real).all()):
            raise ValueError("Means contain NaN/Inf.")
        if not (np.isfinite(sigma_gen).all() and np.isfinite(sigma_real).all()):
            raise ValueError("Covariances contain NaN/Inf.")
        if (
            sigma_gen.shape[0] != sigma_gen.shape[1]
            or sigma_real.shape[0] != sigma_real.shape[1]
        ):
            raise ValueError("Covariances must be square matrices.")

        fid_value = self._calculate_frechet_distance(
            mu_gen, sigma_gen, mu_real, sigma_real
        )

        if not np.isfinite(fid_value):
            raise ValueError("FID computation became unstable and returned non-finite.")

        # 微小负值裁剪（纯数值误差）
        if fid_value < 0 and fid_value > -1e-6:
            fid_value = 0.0

        return float(fid_value)

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
        if not images:
            raise ValueError(
                "At least one image is required to compute Inception Score."
            )

        inception_full = self._get_inception_classifier()

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

        num_images = preds.shape[0]
        effective_splits = min(splits, num_images)
        split_size = num_images // effective_splits

        if split_size == 0:
            raise ValueError(
                f"Inception Score requires at least {splits} images; "
                f"received {num_images}."
            )

        # 计算Inception Score
        split_scores = []
        for i in range(effective_splits):
            part = preds[i * split_size : (i + 1) * split_size, :]

            # 计算KL散度
            kl = part * (np.log(part) - np.log(np.expand_dims(part.mean(axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))

            split_scores.append(np.exp(kl))

        mean_score = float(np.mean(split_scores))
        std_score = float(np.std(split_scores))
        return mean_score, std_score

    def calculate_clip_score(
        self,
        generated_images: List[Image.Image],
        reference_images: List[Image.Image],
        batch_size: int = 32,
    ) -> float:
        """
        计算生成图像与参考图像之间的 CLIP 相似度。

        Args:
            generated_images: 生成图像列表
            reference_images: 真实图像列表
            batch_size: 批处理大小

        Returns:
            CLIPScore（越高越好）
        """
        if len(generated_images) != len(reference_images):
            raise ValueError("Number of generated and reference images must match")

        scores = []

        with torch.no_grad():
            for start in range(0, len(generated_images), batch_size):
                gen_batch = generated_images[start : start + batch_size]
                ref_batch = reference_images[start : start + batch_size]

                gen_batch = [
                    img if img.mode == "RGB" else img.convert("RGB")
                    for img in gen_batch
                ]
                ref_batch = [
                    img if img.mode == "RGB" else img.convert("RGB")
                    for img in ref_batch
                ]

                gen_inputs = self.clip_image_processor(
                    images=gen_batch, return_tensors="pt"
                )
                ref_inputs = self.clip_image_processor(
                    images=ref_batch, return_tensors="pt"
                )

                gen_pixels = gen_inputs["pixel_values"].to(self.device)
                ref_pixels = ref_inputs["pixel_values"].to(self.device)

                gen_features = self.clip_model.get_image_features(
                    pixel_values=gen_pixels
                )
                ref_features = self.clip_model.get_image_features(
                    pixel_values=ref_pixels
                )

                gen_features = F.normalize(gen_features, dim=-1)
                ref_features = F.normalize(ref_features, dim=-1)

                similarity = F.cosine_similarity(gen_features, ref_features)
                scores.extend(similarity.cpu().numpy())

        return float(np.mean(scores) * 100)  # 通常缩放到0-100范围

    def calculate_ssim(
        self, generated_images: List[Image.Image], real_images: List[Image.Image]
    ) -> float:
        """
        计算SSIM（结构相似性指数）

        Args:
            generated_images: 生成的图像列表
            real_images: 真实图像列表

        Returns:
            SSIM分数（越高越好，范围0-1）
        """
        if len(generated_images) != len(real_images):
            raise ValueError("Number of generated and real images must match")

        # 图像预处理 - 统一尺寸并转换为张量
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # 统一尺寸
                transforms.ToTensor(),
            ]
        )

        ssim_scores = []
        with torch.no_grad():
            for gen_img, real_img in zip(generated_images, real_images):
                # 确保图像是RGB模式
                if gen_img.mode != "RGB":
                    gen_img = gen_img.convert("RGB")
                if real_img.mode != "RGB":
                    real_img = real_img.convert("RGB")

                # 转换为张量
                gen_tensor = transform(gen_img).unsqueeze(0).to(self.device)
                real_tensor = transform(real_img).unsqueeze(0).to(self.device)

                # 计算SSIM
                score = ssim(gen_tensor, real_tensor, data_range=1.0, size_average=True)
                ssim_scores.append(score.item())

        return float(np.mean(ssim_scores))

    def calculate_lpips(
        self, generated_images: List[Image.Image], real_images: List[Image.Image]
    ) -> float:
        """
        计算LPIPS（感知路径相似度）使用AlexNet

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

        # 初始化LPIPS模型 - 明确使用AlexNet
        loss_fn = lpips.LPIPS(net="alex").to(self.device)
        logger.info("LPIPS使用AlexNet网络进行计算")

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

        return float(np.mean(distances))

    def _compute_inception_activations(
        self, images: List[Image.Image], batch_size: int
    ) -> np.ndarray:
        """
        使用 Inception 模型抽取图像的特征向量。
        """
        activations = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                processed = []
                for img in batch:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    t = self.inception_transform(img)
                    if t.dtype != torch.float32:
                        t = t.float()
                    processed.append(t)

                batch_tensor = torch.stack(processed, dim=0).to(self.device)
                features = self.inception_model(batch_tensor)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = features.reshape(features.shape[0], -1)
                activations.append(features.cpu().numpy())

        return np.concatenate(activations, axis=0)

    @staticmethod
    def _calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        计算 Frechet 距离（FID）。
        """

        def _to_spd(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
            # 对称化 + 最近 SPD 投影（抬升过小/负特征值）
            S = 0.5 * (S + S.T)
            # eigh 比 eig 更稳，且返回实特征值
            w, v = linalg.eigh(S.astype(np.float64, copy=False))
            w = np.maximum(w, eps)
            return v @ (np.diag(w) @ v.T)

        mu1 = np.atleast_1d(mu1).astype(np.float64, copy=False)
        mu2 = np.atleast_1d(mu2).astype(np.float64, copy=False)
        sigma1 = np.atleast_2d(sigma1).astype(np.float64, copy=False)
        sigma2 = np.atleast_2d(sigma2).astype(np.float64, copy=False)

        # 保证 SPD
        S1 = _to_spd(sigma1, eps)
        S2 = _to_spd(sigma2, eps)

        # sqrtm 及自适应重试（逐步增加 jitter）
        covmean = None
        I = np.eye(S1.shape[0], dtype=np.float64)
        for k in range(4):  # 最多 4 次
            cm_try, _ = linalg.sqrtm(S1 @ S2, disp=False)
            if np.isfinite(cm_try).all():
                covmean = cm_try
                break
            jitter = (10.0**k) * eps
            S1 = _to_spd(S1 + I * jitter, jitter)
            S2 = _to_spd(S2 + I * jitter, jitter)
        if covmean is None or not np.isfinite(covmean).all():
            raise ValueError("sqrtm failed to produce a finite matrix")

        # 复数与对称化处理
        if np.iscomplexobj(covmean):
            imag_max = float(np.max(np.abs(covmean.imag)))
            if imag_max > 1e-3:
                raise ValueError(f"sqrtm produced large imaginary part: {imag_max:.3e}")
            covmean = covmean.real
        covmean = 0.5 * (covmean + covmean.T)

        # 计算 FID
        diff = mu1 - mu2
        fid = float(diff @ diff + np.trace(S1) + np.trace(S2) - 2.0 * np.trace(covmean))

        if not np.isfinite(fid):
            raise ValueError("FID computation returned non-finite value")

        # 微小负值裁剪
        if fid < 0 and fid > -1e-6:
            fid = 0.0
        return fid

    def _get_inception_classifier(self) -> torch.nn.Module:
        """
        懒加载Inception分类模型用于Inception Score计算。
        """
        if self._inception_classifier is None:
            model = inception_v3(weights=self._inception_weights, transform_input=False)
            model.eval()
            model.to(self.device)
            self._inception_classifier = model
        return self._inception_classifier


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

    # 计算SSIM
    try:
        ssim_score = evaluator.calculate_ssim(generated_images, real_images)
        metrics["SSIM"] = ssim_score
        logger.info(f"SSIM: {ssim_score:.4f}")
    except Exception as e:
        logger.warning(f"Failed to calculate SSIM: {e}")

    # 计算LPIPS
    try:
        lpips = evaluator.calculate_lpips(generated_images, real_images)
        metrics["LPIPS"] = lpips
        logger.info(f"LPIPS: {lpips:.4f}")
    except Exception as e:
        logger.warning(f"Failed to calculate LPIPS: {e}")

    return metrics