import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                config["temporal_conv"][0],
                config["temporal_conv"][1],
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
            ),
            nn.BatchNorm2d(config["temporal_conv"][1]),
            nn.ReLU(),
            nn.Conv2d(
                config["temporal_conv"][1],
                config["temporal_conv"][2],
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
            ),
            nn.BatchNorm2d(config["temporal_conv"][2]),
            nn.ReLU(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                config["spatial_conv"][0],
                config["spatial_conv"][1],
                kernel_size=(5, 1),
                stride=(2, 1),
                padding=(2, 0),
            ),
            nn.BatchNorm2d(config["spatial_conv"][1]),
            nn.ReLU(),
            nn.Conv2d(
                config["spatial_conv"][1],
                config["spatial_conv"][2],
                kernel_size=(5, 1),
                stride=(2, 1),
                padding=(2, 0),
            ),
            nn.BatchNorm2d(config["spatial_conv"][2]),
            nn.ReLU(),
        )
        self.ts_conv = nn.Sequential(
            nn.Conv2d(config["ts_conv"][0], config["ts_conv"][1], 3, 2, 1),
            nn.BatchNorm2d(config["ts_conv"][1]),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(config["ts_conv"][1], config["proj_dim"])

        self.class_head = None
        if config["class_head"].get("enabled", False):
            self.class_head = nn.Linear(
                config["proj_dim"], config["class_head"]["n_classes"]
            )

    def forward(self, x, return_class=False):
        """
        前向传播
        Args:
            x: EEG信号 (batch, channels, timepoints)
            return_class: 是否返回类别预测
        Returns:
            h_eeg: EEG embedding (batch, 768)
            class_logits: 类别预测 (可选)
        """
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.ts_conv(x)
        x = x.squeeze()
        h = self.proj(x)
        logits = None
        if self.class_head is not None and return_class:
            logits = self.class_head(h)
        h = F.normalize(h, p=2, dim=1)
        return h, logits


if __name__ == "__main__":
    import torch

    # 测试配置
    config = {
        "temporal_conv": [1, 32, 64],  # [in_channels, mid_channels, out_channels]
        "spatial_conv": [64, 128, 256],  # [in_channels, mid_channels, out_channels]
        "ts_conv": [256, 512],  # [in_channels, out_channels]
        "proj_dim": 768,
        "class_head": {"enabled": True, "n_classes": 40},
    }

    # 创建模型
    model = EEGEncoder(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 创建测试数据 (batch_size=2, channels=1, height=64, width=128)
    # 注意：EEG数据通常形状为 (batch, channels, timepoints)
    # 但在这里，我们使用2D卷积，所以需要4D张量
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 128, 440)

    # 测试前向传播
    with torch.no_grad():
        # 测试不返回类别
        embedding, _ = model(test_input, return_class=False)
        print(f"输出embedding形状: {embedding.shape}")

        # 测试返回类别
        embedding, class_logits = model(test_input, return_class=True)
        print(f"输出embedding形状: {embedding.shape}")
        print(f"输出类别logits形状: {class_logits.shape}")