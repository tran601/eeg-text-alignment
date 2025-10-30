import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CLIPTextEncoder


def normalize_image_key(image_path: str) -> str:
    """Standardize an image path to match entries in the images array."""
    cleaned = image_path.strip().replace("\\", "/")
    cleaned = cleaned.split("/")[-1]
    if "." in cleaned:
        cleaned = cleaned[: cleaned.rfind(".")]
    return cleaned


def flatten_captions(
    caption_payload: Iterable,
) -> List[Tuple[str, List[str]]]:
    """Flatten caption payload into a list of (image_path, captions)."""
    if isinstance(caption_payload, dict):
        return [(path, captions) for path, captions in caption_payload.items()]

    flattened: List[Tuple[str, List[str]]] = []
    for item in caption_payload:
        if isinstance(item, dict):
            for path, captions in item.items():
                flattened.append((path, captions))
    return flattened

def encode_captions(
    text_encoder: CLIPTextEncoder,
    texts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """Encode captions in batches and return an (N, M) tensor."""
    if not texts:
        raise ValueError("No captions provided for encoding.")

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_array = text_encoder.encode_sentence(batch_texts)
        embeddings.append(batch_array)
    return torch.cat(embeddings, dim=0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export caption embeddings.")
    parser.add_argument(
        "--caption-path",
        type=Path,
        default=Path(
            "/home/chengwenjie/datasets/40classes-50images/caption/qwen3vl_multi_caption.json"
        ),
        help="Path to caption JSON file.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/chengwenjie/datasets/40classes-50images/eeg_data/val.pth"),
        help="Path to data file containing the images array.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "/home/chengwenjie/datasets/40classes-50images/embedding/baseline_b32_caption_embeddings.pth"
        ),
        help="Where to store the resulting embeddings payload.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of captions per encoding batch.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not args.caption_path.exists():
        raise FileNotFoundError(f"Caption file not found: {args.caption_path}")

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    data = torch.load(args.data_path)
    images: List[str] = data["images"]
    image_to_idx: Dict[str, int] = {
        normalize_image_key(image): idx for idx, image in enumerate(images)
    }

    with open(args.caption_path, "r", encoding="utf-8") as f:
        captions_raw = json.load(f)

    entries = flatten_captions(captions_raw)
    captions: List[str] = []
    image_paths: List[str] = []
    image_ids: List[int] = []
    skipped: List[str] = []

    for image_path, caption_list in entries:
        normalized = normalize_image_key(image_path)
        image_idx = image_to_idx.get(normalized)
        if image_idx is None:
            skipped.append(image_path)
            continue
        for caption in caption_list:
            if not caption:
                continue
            captions.append(caption)
            image_paths.append(image_path)
            image_ids.append(image_idx)
            break

    if skipped:
        unique_skipped = sorted(set(skipped))
        print(
            f"Warning: {len(unique_skipped)} image paths were not found in the images array."
        )
        for missing in unique_skipped[:10]:
            print(f"  Missing image entry: {missing}")

    text_encoder = CLIPTextEncoder(
        model_path="/home/chengwenjie/workspace/models/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )
    embeddings = encode_captions(text_encoder, captions, args.batch_size).cpu()

    payload = {
        "embeddings": embeddings,
        "captions": captions,
        "image_paths": image_paths,
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)

    print(
        f"Encoded {embeddings.shape[0]} captions with dimension {embeddings.shape[1]}."
    )
    print(f"Payload saved to {args.output_path}")


if __name__ == "__main__":
    main()
