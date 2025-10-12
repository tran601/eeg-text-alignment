# EEG-Text Alignment Project

这是一个用于EEG信号与文本对齐的多模态学习项目，旨在实现从脑电信号到文本的语义对齐，并基于此进行图像生成。

## 项目结构

```
src/
├── dataset/          # 数据处理模块
├── models/           # 模型定义
├── losses/           # 损失函数
├── retrieval/        # 检索模块
├── weighting/        # 加权策略
├── sd/               # Stable Diffusion集成
├── utils/            # 工具函数
├── cli.py            # 命令行接口
├── train_align.py    # 训练脚本
└── test_eval.py      # 评估脚本
```

## 主要功能

- EEG信号编码与特征提取
- 文本编码与语义对齐
- 跨模态检索与匹配
- 基于EEG的图像生成

## 安装与使用

详细的使用说明将在后续完善。

## 许可证

MIT License