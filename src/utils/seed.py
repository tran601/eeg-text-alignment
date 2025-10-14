import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    设置随机种子以确保实验可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 如果使用CUDA，也设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保CUDA操作是确定性的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False