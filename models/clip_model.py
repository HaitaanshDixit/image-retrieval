from typing import List, Union
import torch
import open_clip
from PIL import Image


class ClipEncoder:
    def __init__(self, backbone: str = "ViT-L-14", pretrained: str = "openai",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(backbone)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def zero_shot_classify(self, image: Image.Image, candidate_labels: List[str],
                            template: str = "a photo of a {} garment") -> List[float]:
        image_feat = self.encode_image(image)                     # (1, d)
        prompts = [template.format(lbl) for lbl in candidate_labels]
        text_feats = self.encode_text(prompts)                    # (n, d)
        logits = (100.0 * image_feat @ text_feats.T).softmax(dim=-1)
        return logits.squeeze(0).cpu().tolist()
