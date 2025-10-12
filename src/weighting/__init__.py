"""
加权模块
包含句子级加权和token级融合策略
"""

from .weighting import CaptionWeighting, TokenFusion

__all__ = ["CaptionWeighting", "TokenFusion"]