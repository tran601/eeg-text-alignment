import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from typing import List, Optional, Tuple


class CaptionWeighting:
    """
    Caption加权策略
    实现多种句子级别的加权方法
    """

    @staticmethod
    def sparse_convex_combination(
        query: np.ndarray,
        embeddings: np.ndarray,
        method: str = "nnls",
        ridge: float = 1e-6,
        max_iter: int = 3,
    ) -> np.ndarray:
        """
        稀疏凸组合(SCC)

        Args:
            query: 查询向量 (768,)
            embeddings: 候选向量 (m, 768)
            method: 'nnls', 'fw', 'fw_fc'
            ridge: 正则化系数
            max_iter: Frank-Wolfe迭代次数

        Returns:
            权重向量 (m,)
        """
        m = embeddings.shape[0]

        # 归一化
        query = query / np.linalg.norm(query)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if method == "nnls":
            # Non-negative least squares
            # min ||q - E*w||^2 s.t. w >= 0
            # 添加ridge正则化
            A = embeddings.T  # (768, m)
            A = np.vstack([A, np.sqrt(ridge) * np.eye(m)])  # 添加正则项
            b = np.hstack([query, np.zeros(m)])

            weights, _ = nnls(A, b, maxiter=1000)

        elif method == "fw":
            # Frank-Wolfe算法
            weights = np.zeros(m)
            weights[0] = 1.0  # 初始化

            for _ in range(max_iter):
                # 计算梯度
                residual = query - embeddings.T @ weights
                gradient = -embeddings @ residual

                # 找到最陡下降方向
                idx = np.argmin(gradient)
                s = np.zeros(m)
                s[idx] = 1.0

                # 线搜索
                d = s - weights
                step_size = 2.0 / (_ + 2)  # 经典步长
                weights = weights + step_size * d

        elif method == "fw_fc":
            # Frank-Wolfe with Fully Corrective
            active_set = [0]  # 激活集
            weights = np.zeros(m)
            weights[0] = 1.0

            for _ in range(max_iter):
                # 计算梯度
                residual = query - embeddings.T @ weights
                gradient = -embeddings @ residual

                # 添加新原子到激活集
                idx = np.argmin(gradient)
                if idx not in active_set:
                    active_set.append(idx)

                # 在激活集上做NNLS
                if len(active_set) > 1:
                    E_active = embeddings[active_set]
                    w_active, _ = nnls(E_active.T, query, maxiter=100)

                    # 更新权重
                    weights = np.zeros(m)
                    for i, idx in enumerate(active_set):
                        weights[idx] = w_active[i]

        else:
            raise ValueError(f"Unknown method: {method}")

        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(m) / m

        return weights

    @staticmethod
    def kernel_regression(
        query: np.ndarray,
        embeddings: np.ndarray,
        temperature: float = 0.05,
        top_r: Optional[int] = None,
    ) -> np.ndarray:
        """
        核回归/Softmax加权

        Args:
            query: 查询向量 (768,)
            embeddings: 候选向量 (m, 768)
            temperature: softmax温度
            top_r: 只保留top-r个非零权重

        Returns:
            权重向量 (m,)
        """
        # 归一化
        query = query / np.linalg.norm(query)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 计算相似度
        similarities = embeddings @ query

        # Top-r筛选
        if top_r is not None and top_r < len(similarities):
            top_indices = np.argpartition(similarities, -top_r)[-top_r:]
            mask = np.zeros_like(similarities)
            mask[top_indices] = 1.0
            similarities = similarities * mask - (1 - mask) * 1e10

        # Softmax
        weights = np.exp(similarities / temperature)
        weights = weights / weights.sum()

        return weights

    @staticmethod
    def single_best(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        选择最相似的单个caption

        Args:
            query: 查询向量 (768,)
            embeddings: 候选向量 (m, 768)

        Returns:
            权重向量 (m,) - one-hot
        """
        # 归一化
        query = query / np.linalg.norm(query)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 找到最相似的
        similarities = embeddings @ query
        best_idx = np.argmax(similarities)

        weights = np.zeros(len(embeddings))
        weights[best_idx] = 1.0

        return weights


class TokenFusion:
    """
    Token级别的融合策略
    将句子级权重转换为token级别的融合
    """

    @staticmethod
    def broadcast_fusion(
        token_matrices: List[np.ndarray], sentence_weights: np.ndarray
    ) -> np.ndarray:
        """
        广播融合 - 简单加权平均

        Args:
            token_matrices: 每个caption的token矩阵列表 [(77, 768), ...]
            sentence_weights: 句子权重 (m,)

        Returns:
            融合后的token矩阵 (77, 768)
        """
        # 加权求和
        result = np.zeros_like(token_matrices[0])
        for mat, w in zip(token_matrices, sentence_weights):
            result += w * mat

        # 逐行归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms

        return result

    @staticmethod
    def token_aware_fusion(
        token_matrices: List[np.ndarray],
        sentence_weights: np.ndarray,
        query_embedding: np.ndarray,
        temperature: float = 0.05,
    ) -> np.ndarray:
        """
        Token感知融合

        Args:
            token_matrices: token矩阵列表
            sentence_weights: 句子权重
            query_embedding: EEG查询向量
            temperature: softmax温度

        Returns:
            融合后的token矩阵 (77, 768)
        """
        n_tokens = token_matrices[0].shape[0]
        result = np.zeros_like(token_matrices[0])

        query_norm = query_embedding / np.linalg.norm(query_embedding)

        for t in range(n_tokens):
            # 收集所有caption在位置t的token
            tokens_t = np.stack([mat[t] for mat in token_matrices])

            # 归一化
            tokens_t_norm = tokens_t / np.linalg.norm(tokens_t, axis=1, keepdims=True)

            # 计算与query的相似度
            token_sims = tokens_t_norm @ query_norm

            # Token级别的attention权重
            token_weights = np.exp(token_sims / temperature)
            token_weights = token_weights / token_weights.sum()

            # 结合句子权重
            final_weights = sentence_weights * token_weights
            final_weights = final_weights / final_weights.sum()

            # 加权融合
            result[t] = np.sum(tokens_t * final_weights[:, np.newaxis], axis=0)

        # 逐行归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms

        return result

    @staticmethod
    def base_align_fusion(
        token_matrices: List[np.ndarray],
        sentence_weights: np.ndarray,
        query_embedding: np.ndarray,
        tau_align: Tuple[float, float] = (0.08, 0.08),
        alpha_minmax: Tuple[float, float] = (0.3, 0.7),
    ) -> np.ndarray:
        """
        基准对齐融合
        以权重最大的caption为基准，进行软对齐

        Args:
            token_matrices: token矩阵列表
            sentence_weights: 句子权重
            query_embedding: EEG查询向量
            tau_align: 对齐温度参数
            alpha_minmax: 门控参数范围

        Returns:
            融合后的token矩阵 (77, 768)
        """
        # 找到基准caption（权重最大的）
        base_idx = np.argmax(sentence_weights)
        base_matrix = token_matrices[base_idx]
        n_tokens = base_matrix.shape[0]

        result = base_matrix.copy()
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        for t in range(n_tokens):
            base_token = base_matrix[t]
            base_norm = base_token / np.linalg.norm(base_token)

            # 跳过特殊token
            if t == 0 or t == n_tokens - 1:  # [SOS], [EOS]
                continue

            # 检查是否是PAD
            if np.linalg.norm(base_token) < 0.1:
                continue

            # 收集其他caption的匹配token
            aligned_tokens = []
            aligned_weights = []

            for i, mat in enumerate(token_matrices):
                if i == base_idx:
                    continue

                # 计算与基准token的相似度
                mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
                base_sims = mat_norm @ base_norm

                # 计算与query的相似度
                query_sims = mat_norm @ query_norm

                # 综合相似度
                combined_sims = base_sims * tau_align[0] + query_sims * tau_align[1]

                # 软匹配
                match_weights = np.exp(combined_sims / 0.05)
                match_weights = match_weights / match_weights.sum()

                # 加权平均得到对齐的token
                aligned_token = np.sum(mat * match_weights[:, np.newaxis], axis=0)
                aligned_tokens.append(aligned_token)
                aligned_weights.append(sentence_weights[i])

            if aligned_tokens:
                # 计算匹配分布的熵（用于自适应门控）
                entropy = -np.sum(aligned_weights * np.log(aligned_weights + 1e-10))
                max_entropy = np.log(len(aligned_weights))

                # 自适应门控参数
                alpha = alpha_minmax[0] + (alpha_minmax[1] - alpha_minmax[0]) * (
                    entropy / max_entropy
                )

                # 融合
                aligned_avg = np.average(
                    aligned_tokens, weights=aligned_weights, axis=0
                )
                result[t] = (1 - alpha) * base_token + alpha * aligned_avg

        # 逐行归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms

        return result