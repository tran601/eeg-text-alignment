import torch
import yaml
from pathlib import Path


class _ConfigDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    style = '"' if data in {"1e-8"} else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ConfigDumper.add_representer(str, _str_representer)

path = "checkpoints/LowLevel/SupConCrossModalLoss/20251028_0911_sup_with_proj_head_512_split_by_image_id_b32_sample8_prototype/checkpoint_step_25000_noise_std_1.pt"

checkpoint = torch.load(path, map_location="cpu")
config = checkpoint["config"]

output_path = Path("configs/default.yaml")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as f:
    yaml.dump(config, f, Dumper=_ConfigDumper, sort_keys=False)

print(f"Config restored to {output_path}")