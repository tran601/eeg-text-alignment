import numpy as np
import torch
import faiss
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import pickle


@dataclass
class RetrievalHit:
    """检索结果"""

    caption_id: int
    image_id: int
    embedding: np.ndarray
    similarity: float
    caption_text: Optional[str] = None


class CaptionRetriever:
    """
    全库Caption检索器
    实现CATVis论文中的检索和重排策略
    """

    def __init__(self, config: dict):
        self.config = config
        self.index = None
        self.caption_embeddings = None
        self.caption_metadata = None  # {caption_id: {'image_id': ..., 'text': ...}}

    def build_index(self, embeddings: np.ndarray, metadata: Dict):
        """
        构建FAISS索引

        Args:
            embeddings: caption句向量 (N, 768)
            metadata: caption元数据
        """
        self.caption_embeddings = embeddings
        self.caption_metadata = metadata

        # L2归一化
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 构建FAISS索引
        d = embeddings.shape[1]

        if self.config["index"] == "faiss_flat":
            # 精确搜索
            self.index = faiss.IndexFlatIP(d)  # 内积 = 余弦相似度（归一化后）
        elif self.config["index"] == "faiss_ivf_pq":
            # 近似搜索，更快
            quantizer = faiss.IndexFlatIP(d)
            n_list = min(100, len(embeddings) // 100)  # 聚类中心数
            m = 32  # PQ子空间数
            self.index = faiss.IndexIVFPQ(quantizer, d, n_list, m, 8)
            self.index.train(embeddings.astype(np.float32))
        else:
            raise ValueError(f"Unknown index type: {self.config['index']}")

        self.index.add(embeddings.astype(np.float32))

    def search_topk(self, query_embedding: np.ndarray, k: int) -> List[RetrievalHit]:
        """
        检索top-k最相似的captions

        Args:
            query_embedding: 查询向量 (768,)
            k: 返回top-k个结果

        Returns:
            检索结果列表
        """
        # 归一化查询向量
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # FAISS搜索
        similarities, indices = self.index.search(query_embedding, k)

        # 构建结果
        hits = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < 0:  # FAISS可能返回-1表示无效结果
                continue

            hit = RetrievalHit(
                caption_id=int(idx),
                image_id=self.caption_metadata[idx]["image_id"],
                embedding=self.caption_embeddings[idx],
                similarity=float(sim),
                caption_text=self.caption_metadata[idx].get("text", ""),
            )
            hits.append(hit)

        return hits

    def rerank_by_class(
        self,
        hits: List[RetrievalHit],
        eeg_embedding: np.ndarray,
        class_embedding: np.ndarray,
        gamma: float = 0.3,
        top_r: int = 3,
    ) -> List[RetrievalHit]:
        """
        基于类一致性重排
        实现CATVis论文3.3节的重排策略

        Args:
            hits: 初始检索结果
            eeg_embedding: EEG查询向量
            class_embedding: 预测类别的CLIP嵌入
            gamma: 类一致性权重
            top_r: 最终返回的caption数量

        Returns:
            重排后的top-r结果
        """
        # 按image_id分组
        image_groups = {}
        for hit in hits:
            if hit.image_id not in image_groups:
                image_groups[hit.image_id] = []
            image_groups[hit.image_id].append(hit)

        # 计算每组的分数
        group_scores = []
        for image_id, group_hits in image_groups.items():
            # 提取组内所有caption的embeddings
            group_embeds = np.stack([h.embedding for h in group_hits])

            # 归一化
            group_embeds = group_embeds / np.linalg.norm(
                group_embeds, axis=1, keepdims=True
            )
            eeg_norm = eeg_embedding / np.linalg.norm(eeg_embedding)

            # 计算与EEG的最大相似度（max-pooling）
            eeg_sims = np.dot(group_embeds, eeg_norm)
            max_sim = np.max(eeg_sims)

            # 计算组的平均embedding
            mean_embed = np.mean(group_embeds, axis=0)
            mean_embed = mean_embed / np.linalg.norm(mean_embed)

            # 计算与类别的相似度
            class_norm = class_embedding / np.linalg.norm(class_embedding)
            class_sim = np.dot(mean_embed, class_norm)

            # 综合得分
            score = max_sim + gamma * class_sim

            group_scores.append(
                {"image_id": image_id, "score": score, "hits": group_hits}
            )

        # 排序
        group_scores.sort(key=lambda x: x["score"], reverse=True)

        # 选择得分最高的组，返回前r个caption
        if group_scores:
            best_group = group_scores[0]["hits"]
            # 在组内按与EEG的相似度排序
            best_group.sort(key=lambda x: x.similarity, reverse=True)
            return best_group[:top_r]

        return hits[:top_r]

    def save_index(self, path: str):
        """保存索引到文件"""
        data = {
            "index": faiss.serialize_index(self.index),
            "embeddings": self.caption_embeddings,
            "metadata": self.caption_metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, path: str):
        """从文件加载索引"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data["index"])
        self.caption_embeddings = data["embeddings"]
        self.caption_metadata = data["metadata"]