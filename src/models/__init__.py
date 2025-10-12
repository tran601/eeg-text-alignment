"""
模型模块
包含EEG编码器和文本编码器
"""

from .eeg_encoder import EEGEncoder
from .text_encoder import CLIPTextEncoder

__all__ = ["EEGEncoder", "CLIPTextEncoder"]