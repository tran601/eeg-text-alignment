import torch
import numpy as np

from transformers import CLIPModel, CLIPTokenizerFast


class CLIPTextEncoder:
    """
    CLIP文本编码器
    """

    def __init__(
        self,
        model_path: str = "/home/chengwenjie/workspace/models/clip-vit-large-patch14",
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        初始化CLIP文本编码器

        Args:
            model_path: 模型目录
            torch_dtype: 数据类型
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype

        self.text_encoder = CLIPModel.from_pretrained(model_path)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_path)

        self.text_encoder = self.text_encoder.to(self.device, dtype=self.torch_dtype)
        self.text_encoder.eval()

    @torch.no_grad()
    def encode_sentence(self, text) -> np.ndarray:
        """
        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            句子向量
        """

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # EOT pool
        sentence_vector = self.text_encoder.get_text_features(**inputs)

        return sentence_vector


if __name__ == "__main__":
    text_encoder = CLIPTextEncoder()
    texts = ["hello", "hello world"]
    out = text_encoder.encode_sentence(text=texts)
    print(out)