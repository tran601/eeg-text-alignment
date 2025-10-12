import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SupConCrossModalLoss(nn.Module):
    """
    跨模态 Supervised Contrastive
    - 把 EEG 与 Text 合并，按 image_id 作为"类"；
    - 默认仅将异模态同类作为正样本（cross_only=False），可切换包含同模态正样本；
    - 损失：对每个锚点，正样本均匀平均，分母包含所有非自身的样本。
    - 注：为简洁起见不使用 memory bank（可扩展为双 bank）。
    """

    def __init__(self, config):
        super().__init__()
        self.temperature = float(config["temperature"])
        self.cross_only = bool(config["cross_only"])
        self.l2_normalize = bool(config["l2_normalize"])

    def forward(
        self,
        eeg_embeddings: torch.Tensor,  # [Ne, D]
        text_embeddings: torch.Tensor,  # [Nt, D]
        eeg_image_ids: torch.Tensor,  # [Ne]
        text_image_ids: torch.Tensor,  # [Nt]
    ) -> torch.Tensor:
        device = eeg_embeddings.device
        B, D = eeg_embeddings.shape

        if self.l2_normalize:
            eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        if eeg_image_ids is None:
            eeg_image_ids_tensor = torch.arange(B, device=device, dtype=torch.long)
        else:
            eeg_image_ids_tensor = eeg_image_ids.to(device)

        # 合并池 combined_embeddings 与标签 combined_image_ids
        combined_embeddings = torch.cat(
            [eeg_embeddings, text_embeddings], dim=0
        )  # [N, D], N=B+N_text
        combined_image_ids = torch.cat(
            [eeg_image_ids_tensor, text_image_ids], dim=0
        )  # [N]
        N = combined_embeddings.size(0)

        # 模态掩码（用于 cross_only）
        is_eeg_modality = torch.zeros(N, dtype=torch.bool, device=device)
        is_eeg_modality[:B] = True
        is_text_modality = ~is_eeg_modality

        # 相似度与 logits
        similarity = combined_embeddings @ combined_embeddings.t()  # [N, N]
        logits = similarity / self.temperature  # [N, N]

        # 排除自身（对角）参与分母
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        logits = logits.masked_fill(self_mask, float("-inf"))

        # 正样本掩码：同 image_id
        same_image_mask = (
            combined_image_ids.view(-1, 1) == combined_image_ids.view(1, -1)
        ) & (~self_mask)
        if self.cross_only:
            cross_modality_mask = (
                is_eeg_modality.unsqueeze(1) & is_text_modality.unsqueeze(0)
            ) | (is_text_modality.unsqueeze(1) & is_eeg_modality.unsqueeze(0))
            positive_mask = same_image_mask & cross_modality_mask
        else:
            positive_mask = same_image_mask

        positive_count = positive_mask.sum(dim=1)  # [N]
        valid_anchor_mask = positive_count > 0

        if not valid_anchor_mask.any():
            return combined_embeddings.new_tensor(0.0)

        # log_prob = log( exp(sim)/sum_{z != a} exp(sim) )
        denominator = torch.logsumexp(logits, dim=1, keepdim=True)  # [N,1]
        log_probabilities = logits - denominator  # [N,N]

        sum_log_prob_positive = (log_probabilities * positive_mask.float()).sum(
            dim=1
        )  # [N]
        per_anchor_loss = -sum_log_prob_positive / positive_count.clamp_min(1)  # [N]

        return per_anchor_loss[valid_anchor_mask].mean()


class MPNCELoss(nn.Module):
    """
    MPNCE 损失封装：
    - 提供单向 mpnce 和对称 symmetric 两种计算方式
    - 两种方法中相同语义变量名保持一致

    参数:
        temperature (float): 温度系数
        average_positive (bool): True=对每个正样本分别取 log 再平均；False=分子为正样本得分指数和
        epsilon (float): 数值稳定用的极小值
        l2_normalize (bool): 在 symmetric 中对输入嵌入进行 L2 归一化
    """

    def __init__(self, config):
        super().__init__()
        self.temperature = float(config["temperature"])
        self.average_positive = bool(config["average_positive"])
        self.epsilon = float(config["epsilon"])
        self.l2_normalize = bool(config["l2_normalize"])

    def mpnce(
        self,
        similarity: torch.Tensor,  # [Ne, Nt] = EEG 与 Text 的相似度矩阵（通常为 L2 归一化后的点积）
        positive_mask: torch.Tensor,  # [Ne, Nt] bool，同图像为 True 的正样本掩码
    ) -> torch.Tensor:
        """
        单向 MPNCE 损失（EEG->Text 或 Text->EEG 任一方向的矩阵）
        """
        # 归一化温度并做数值稳定平移
        logits = similarity / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values

        exp_logits = torch.exp(logits)  # [Ne, Nt]
        denominator = exp_logits.sum(dim=1, keepdim=True) + self.epsilon

        positive_mask_f = positive_mask.float()
        positive_count = positive_mask_f.sum(dim=1) + self.epsilon  # [Ne]

        if self.average_positive:
            # 变体2：对每个正样本分别取 log 再平均
            log_probabilities = logits - torch.log(denominator)  # [Ne, Nt]
            loss_per_row = (
                -(positive_mask_f * log_probabilities).sum(dim=1) / positive_count
            )
            return loss_per_row.mean()
        else:
            # 变体1：分子为正样本 score 的指数和
            numerator = (exp_logits * positive_mask_f).sum(dim=1) + self.epsilon
            loss_per_row = -torch.log(numerator / denominator.squeeze(1))
            return loss_per_row.mean()

    def forward(
        self,
        eeg_embeddings: torch.Tensor,  # [Ne, d]
        text_embeddings: torch.Tensor,  # [Nt, d]
        eeg_image_ids: torch.Tensor,  # [Ne]
        text_image_ids: torch.Tensor,  # [Nt]
    ) -> torch.Tensor:
        """
        对称 MPNCE 损失 = EEG->Text 与 Text->EEG 两个方向之和
        - 如 l2_normalize=True，会对输入的嵌入做 L2 归一化
        """
        if self.l2_normalize:
            eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        # 相似度
        similarity = eeg_embeddings @ text_embeddings.t()  # [Ne, Nt]

        # 正样本掩码
        positive_mask_e2t = (
            eeg_image_ids[:, None] == text_image_ids[None, :]
        )  # [Ne, Nt]
        positive_mask_t2e = positive_mask_e2t.t()  # [Nt, Ne]

        # 两个方向的 mpnce
        loss_e2t = self.mpnce(similarity, positive_mask_e2t)
        loss_t2e = self.mpnce(similarity.t(), positive_mask_t2e)

        return loss_e2t + loss_t2e