"""
损失函数模块
包含Multi-Positive InfoNCE损失
"""

from .mp_infonce import MPNCELoss, SupConCrossModalLoss

__all__ = ["MPNCELoss", "SupConCrossModalLoss"]