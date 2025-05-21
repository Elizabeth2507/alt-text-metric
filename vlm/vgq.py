import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import List, Tuple


class SmolVLMEmbedder:
    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)

    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        patch_embeds, _ = self.get_image_embedding_with_cls(image)
        return patch_embeds

    def get_image_embedding_with_cls(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device, torch.bfloat16)
        pixel_values = inputs.pixel_values

        if pixel_values.ndim == 5:
            pixel_values = pixel_values[:, 0]

        with torch.no_grad():
            vision_outputs = self.model.model.vision_model(pixel_values=pixel_values)
            hidden_states = vision_outputs.last_hidden_state  # (1, N, D)
            projected = self.model.model.connector(hidden_states)  # (1, N, D)

        projected = projected.squeeze(0)  # (N, D)
        cls_embedding = projected[0]      # [CLS] token assumed at position 0
        return projected, cls_embedding  # (N, D), (D,)

    def get_text_embeddings(self, text: str):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device, torch.bfloat16)

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
                output_hidden_states=True
            )

        tokens = self.processor.tokenizer.tokenize(text)
        text_hidden_states = outputs.last_hidden_state

        return tokens, text_hidden_states.squeeze(0).cpu()


class VGQScorer:
    def __init__(self, embedder: SmolVLMEmbedder, use_cls_context: bool = False):
        self.embedder = embedder
        self.use_cls_context = use_cls_context

        dummy_image = Image.new("RGB", (224, 224))
        region_embeds, _ = self.embedder.get_image_embedding_with_cls(dummy_image)
        self.region_dim = region_embeds.shape[-1]

        dummy_tokens, dummy_text_embeds = self.embedder.get_text_embeddings("dummy")
        self.text_dim = dummy_text_embeds.shape[-1]

        # Add a fusion layer if using CLS context
        if use_cls_context:
            self.fusion_layer = nn.Linear(self.region_dim * 2, self.region_dim)

        # Text projection: align text embedding space to vision
        self.text_proj = nn.Linear(self.text_dim, self.region_dim)

    def encode_regions(
        self,
        region_crops: List[Image.Image],
        full_image: Image.Image = None
    ) -> torch.Tensor:
        region_embeddings = []

        # Optional CLS token from full image
        global_cls = None
        if self.use_cls_context:
            if full_image is None:
                raise ValueError("full_image must be provided when use_cls_context=True")
            _, global_cls = self.embedder.get_image_embedding_with_cls(full_image)
            global_cls = global_cls.float()

        for region in region_crops:
            patch_embeds, _ = self.embedder.get_image_embedding_with_cls(region)
            pooled = patch_embeds.mean(dim=0).float()

            if self.use_cls_context:
                fused = torch.cat([pooled, global_cls], dim=0)  # (2D,)
                fused = self.fusion_layer(fused)  # Reduce to (D,)
            else:
                fused = pooled

            region_embeddings.append(fused)

        return torch.stack(region_embeddings)  # (R, D)


    def match_tokens_to_regions_cosine(
    self,
    text: str,
    region_crops: List[Image.Image],
    full_image: Image.Image = None
    ) -> Tuple[List[str], torch.Tensor, List[torch.Tensor]]:
        tokens, text_embeds = self.embedder.get_text_embeddings(text)  # (T, D_text)
        text_embeds = self.text_proj(text_embeds.float())  # (T, D_region)
        region_patches = self.encode_regions(region_crops, full_image)  # (R, D_region)

        similarity_matrix = []

        for token_vec in text_embeds:  # (D,)
            token_vec = F.normalize(token_vec, dim=-1)
            region_scores = []
            for patches in region_patches:
                patches = F.normalize(patches, dim=-1)
                sims = F.cosine_similarity(patches, token_vec.unsqueeze(0), dim=-1)
                region_scores.append(sims.max().item())
            similarity_matrix.append(region_scores)

        return tokens, torch.tensor(similarity_matrix), region_patches

    def get_token_region_matches(
        self, text: str, region_crops: List[Image.Image], full_image: Image.Image = None, threshold: float = 0.4
    ) -> List[dict]:
        tokens, similarity, _ = self.match_tokens_to_regions_cosine(text, region_crops, full_image)
        max_scores, best_regions = similarity.max(dim=1)

        matches = []
        for i, token in enumerate(tokens):
            region_id = best_regions[i].item()
            score = max_scores[i].item()
            match = {
                "token": token,
                "region_id": region_id,
                "score": score,
                "matched": score >= threshold,
            }
            matches.append(match)
        return matches
    