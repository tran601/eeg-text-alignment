import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保CUDA操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")