# EEG-Text Alignment Project

这是一个用于EEG信号与文本对齐的多模态学习项目，旨在实现从脑电信号到文本的语义对齐，并基于此进行图像生成。

## 项目概述

本项目实现了以下核心功能：
- EEG信号编码与特征提取
- 文本编码与语义对齐
- 跨模态检索与匹配
- 基于EEG的图像生成

## 项目结构

```
src/
├── dataset/          # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py        # EEG数据集类
│   ├── preprocess.py     # 数据预处理
│   └── splits.py         # 数据集分割
├── models/           # 模型定义
│   ├── __init__.py
│   ├── eeg_encoder.py    # EEG编码器
│   └── text_encoder.py   # CLIP文本编码器
├── losses/           # 损失函数
│   ├── __init__.py
│   └── mp_infonce.py     # Multi-Positive InfoNCE损失
├── retrieval/        # 检索模块
│   ├── __init__.py
│   ├── retrieval.py       # 文本检索器
│   ├── index.py           # 索引构建
│   └── rerank.py          # 重排策略
├── weighting/        # 加权策略
│   ├── __init__.py
│   ├── weighting.py       # 句子级加权和token级融合
│   ├── kernel_reg.py      # 核回归
│   ├── scc_fw.py          # 稀疏凸组合Frank-Wolfe
│   └── token_fusion.py    # Token级融合
├── sd/               # Stable Diffusion集成
│   ├── __init__.py
│   └── pipeline.py        # EEG到图像生成管道
├── utils/            # 工具函数
│   ├── __init__.py
│   ├── metrics.py         # 评估指标
│   ├── seed.py            # 随机种子设置
│   └── logging.py         # 日志工具
├── cli.py            # 命令行接口
├── train_align.py    # 训练脚本
└── test_eval.py      # 评估脚本
```

## 主要功能

### 1. EEG编码器
- 实现了基于卷积神经网络的EEG信号编码器
- 支持时序和空间特征提取
- 可选的分类头用于EEG信号分类

### 2. 文本编码器
- 基于CLIP的文本编码器
- 支持句子级和token级编码
- 集成缓存机制提高效率

### 3. 跨模态对齐
- 实现了Multi-Positive InfoNCE损失函数
- 支持跨模态监督对比学习
- 多种加权策略和融合方法

### 4. 图像生成
- 集成Stable Diffusion进行图像生成
- 基于EEG信号的条件生成
- 支持文本检索和重排

## 安装与使用

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (推荐)
- 其他依赖见 requirements.txt

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/tran601/eeg-text-alignment.git
cd eeg-text-alignment
```

2. 创建虚拟环境
```bash
conda create -n eeg-text python=3.8
conda activate eeg-text
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

### 使用方法

#### 数据准备
```bash
python src/cli.py prepare --config configs/default.yaml
```

#### 训练模型
```bash
python src/cli.py train --config configs/default.yaml
```

#### 评估模型
```bash
python src/cli.py evaluate --config configs/default.yaml --checkpoint path/to/checkpoint.pt
```

#### 生成图像
```bash
python src/cli.py generate --config configs/default.yaml --checkpoint path/to/checkpoint.pt --num_samples 10
```

## 配置说明

项目使用YAML格式的配置文件，主要包含以下部分：
- `model`: 模型配置
- `data`: 数据配置
- `train`: 训练配置
- `eval`: 评估配置
- `sd`: Stable Diffusion配置

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 引用

如果您使用了本项目的代码，请考虑引用：

```bibtex
@misc{eeg-text-alignment,
  title={EEG-Text Alignment Project},
  author={EEG2Text Team},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/tran601/eeg-text-alignment}}
}
```