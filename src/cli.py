#!/usr/bin/env python3
"""
EEG到文本对齐与条件生成项目的命令行接口
提供数据准备、训练、验证和测试的统一入口
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.train_align import AlignmentTrainer
from src.test_eval import Evaluator
from src.data.preprocess import DataPreprocessor
from src.scripts.build_text_index import build_text_index
from src.scripts.prepare_dataset import prepare_dataset

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: dict = None) -> dict:
    """
    加载配置文件并应用覆盖

    Args:
        config_path: 配置文件路径
        overrides: 覆盖参数字典

    Returns:
        合并后的配置字典
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 应用覆盖参数
    if overrides:

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(config, overrides)

    return config


def prepare_data(args):
    """准备数据集"""
    logger.info("Preparing dataset...")

    # 加载配置
    config = load_config(args.config)

    # 创建数据预处理器
    preprocessor = DataPreprocessor(config)

    # 执行预处理
    preprocessor.prepare_all()

    logger.info("Dataset preparation completed!")


def build_index(args):
    """构建文本索引"""
    logger.info("Building text index...")

    # 加载配置
    config = load_config(args.config)

    # 构建索引
    build_text_index(config)

    logger.info("Text index building completed!")


def train(args):
    """训练模型"""
    logger.info("Starting training...")

    # 加载配置
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            # 尝试解析为正确的类型
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # 保持字符串

            # 处理嵌套键，如 "model.eeg_encoder.d_model=512"
            keys = key.split(".")
            current = overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    config = load_config(args.config, overrides)

    # 创建训练器
    trainer = AlignmentTrainer(args.config)

    # 开始训练
    trainer.train()

    logger.info("Training completed!")


def evaluate(args):
    """评估模型"""
    logger.info("Starting evaluation...")

    # 加载配置
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            keys = key.split(".")
            current = overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    config = load_config(args.config, overrides)

    # 创建评估器
    evaluator = Evaluator(config)

    # 执行评估
    results = evaluator.evaluate()

    # 打印结果
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # 保存结果
    if args.output:
        with open(args.output, "w") as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {args.output}")

    logger.info("Evaluation completed!")


def generate(args):
    """生成图像"""
    logger.info("Starting image generation...")

    # 加载配置
    config = load_config(args.config)

    # 这里可以添加生成逻辑
    # TODO: 实现图像生成功能

    logger.info("Image generation completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="EEG-Text Alignment and Generation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 准备数据命令
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )

    # 构建索引命令
    index_parser = subparsers.add_parser("build-index", help="Build text index")
    index_parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )

    # 训练命令
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    train_parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override config parameters (key=value)",
    )

    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    eval_parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override config parameters (key=value)",
    )
    eval_parser.add_argument(
        "--output", type=str, default="results.yaml", help="Output file for results"
    )

    # 生成命令
    gen_parser = subparsers.add_parser("generate", help="Generate images")
    gen_parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    gen_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default="generated_images",
        help="Output directory for generated images",
    )
    gen_parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )

    # 解析参数
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 执行相应命令
    try:
        if args.command == "prepare":
            prepare_data(args)
        elif args.command == "build-index":
            build_index(args)
        elif args.command == "train":
            train(args)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "generate":
            generate(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
    except Exception as e:
        logger.error(f"Error executing {args.command}: {e}")
        raise


if __name__ == "__main__":
    main()