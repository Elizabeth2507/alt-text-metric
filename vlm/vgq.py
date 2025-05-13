# import torch
# import torch.nn.functional as F
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
# from typing import List, Tuple


# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image

# class SmolVLMEmbedder:
#     def __init__(self, model_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.processor = AutoProcessor.from_pretrained(model_path)
#         self.model = AutoModelForImageTextToText.from_pretrained(
#             model_path, torch_dtype=torch.bfloat16
#         ).to(self.device)

#     def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
#         inputs = self.processor(images=[image], return_tensors="pt").to(self.device, torch.bfloat16)
#         pixel_values = inputs.pixel_values

#         if pixel_values.ndim == 5:
#             pixel_values = pixel_values[:, 0]

#         with torch.no_grad():
#             vision_outputs = self.model.model.vision_model(pixel_values=pixel_values)
#             image_hidden_states = vision_outputs.last_hidden_state
#             image_hidden_states = self.model.model.connector(image_hidden_states)

#         return image_hidden_states.squeeze(0).cpu()

#     def get_text_embeddings(self, text: str):
#         inputs = self.processor(text=text, return_tensors="pt").to(self.device, torch.bfloat16)

#         with torch.no_grad():
#             outputs = self.model.model(
#                 input_ids=inputs.input_ids,
#                 attention_mask=inputs.attention_mask,
#                 return_dict=True,
#                 output_hidden_states=True
#             )

#         tokens = self.processor.tokenizer.tokenize(text)
#         text_hidden_states = outputs.last_hidden_state

#         return tokens, text_hidden_states.squeeze(0).cpu()
    

# class VGQScorer:
#     def __init__(self, embedder: SmolVLMEmbedder, use_cls_context: bool = False):
#         self.embedder = embedder
#         self.use_cls_context = use_cls_context
    
#     def encode_regions(
#         self,
#         region_crops: List[Image.Image],
#         full_image: Image.Image = None
#     ) -> torch.Tensor:
#         region_embeddings = []

#         if self.use_cls_context:
#             assert full_image is not None, "Full image is required for CLS-based region encoding."
#             _, global_cls = self.embedder.get_image_embedding_with_cls(full_image)  # (D,)
#         else:
#             global_cls = None

#         for region in region_crops:
#             patch_embeds, _ = self.embedder.get_image_embedding_with_cls(region)
#             pooled = patch_embeds.mean(dim=0)  # (D,)

#             if self.use_cls_context:
#                 # Concatenate CLS to pooled region vector (D + D → 2D)
#                 fused = torch.cat([pooled, global_cls], dim=0)
#             else:
#                 fused = pooled

#             region_embeddings.append(fused)

#         return torch.stack(region_embeddings)  # (R, D) or (R, 2D)


#     # def encode_regions(self, region_crops: List[Image.Image]) -> torch.Tensor:
#     #     region_embeddings = []
#     #     for region in region_crops:
#     #         image_embed = self.embedder.get_image_embedding(region)
#     #         pooled = image_embed.mean(dim=0)  # Pool to (dim,)
#     #         region_embeddings.append(pooled)
#     #     return torch.stack(region_embeddings)  # (R, D)

#     def match_tokens_to_regions_cosine(
#         self,
#         text: str,
#         region_crops: List[Image.Image],
#     ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
#         tokens, text_embeds = self.embedder.get_text_embeddings(text)
#         region_embeds = self.encode_regions(region_crops)

#         # Normalize
#         text_norm = F.normalize(text_embeds, dim=-1)         # (T, D)
#         region_norm = F.normalize(region_embeds, dim=-1)     # (R, D)

#         # Similarity matrix: (T, R)
#         similarity = torch.matmul(text_norm, region_norm.T)

#         return tokens, similarity, region_embeds
    
#     def match_tokens_to_regions_softmax(
#         self,
#         text: str,
#         region_crops: List[Image.Image],
#         temperature: float = 0.07
#     ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
#         """
#         Computes softmax-normalized dot-product similarity between text tokens and image regions.
#         This is an alternative to cosine similarity for sharper contrastive alignment.
#         """
#         tokens, text_embeds = self.embedder.get_text_embeddings(text)
#         region_embeds = self.encode_regions(region_crops)

#         # Compute raw dot-product similarity
#         similarity = torch.matmul(text_embeds, region_embeds.T)  # (T, R)

#         # Temperature scaling + softmax
#         similarity_scaled = similarity / temperature
#         similarity_softmax = torch.softmax(similarity_scaled, dim=1)

#         return tokens, similarity_softmax, region_embeds
    
#     def match_tokens_to_regions_softmax_with_raw(
#         self,
#         text: str,
#         region_crops: List[Image.Image],
#         temperature: float = 0.07
#     ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
#         tokens, text_embeds = self.embedder.get_text_embeddings(text)
#         region_embeds = self.encode_regions(region_crops)

#         similarity = torch.matmul(text_embeds, region_embeds.T)
#         similarity_scaled = similarity / temperature
#         similarity_softmax = torch.softmax(similarity_scaled, dim=1)

#         return tokens, similarity, similarity_softmax, region_embeds



#     def get_token_region_matches(
#         self, text: str, region_crops: List[Image.Image], threshold: float = 0.4
#     ) -> List[dict]:
#         tokens, similarity, _ = self.match_tokens_to_regions_softmax(text, region_crops)
#         # tokens, raw_similarity, softmax_similarity, region_embeds = self.match_tokens_to_regions_softmax_with_raw(text, region_crops)
#         max_scores, best_regions = raw_similarity.max(dim=1)

#         matches = []
#         for i, token in enumerate(tokens):
#             region_id = best_regions[i].item()
#             score = max_scores[i].item()
#             match = {
#                 "token": token,
#                 "region_id": region_id,
#                 "score": score,
#                 "matched": score >= threshold,
#             }
#             matches.append(match)
#         return matches


import torch
import torch.nn.functional as F
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

    def encode_regions(
        self,
        region_crops: List[Image.Image],
        full_image: Image.Image = None
    ) -> torch.Tensor:
        region_embeddings = []

        if self.use_cls_context:
            if full_image is None:
                raise ValueError("full_image must be provided when use_cls_context=True")
            _, global_cls = self.embedder.get_image_embedding_with_cls(full_image)
        else:
            global_cls = None

        for region in region_crops:
            patch_embeds, _ = self.embedder.get_image_embedding_with_cls(region)
            pooled = patch_embeds.mean(dim=0)

            if self.use_cls_context:
                fused = torch.cat([pooled, global_cls], dim=0)
            else:
                fused = pooled

            region_embeddings.append(fused)

        return torch.stack(region_embeddings)  # (R, D) or (R, 2D)

    def match_tokens_to_regions_cosine(
        self,
        text: str,
        region_crops: List[Image.Image],
        full_image: Image.Image = None
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        tokens, text_embeds = self.embedder.get_text_embeddings(text)
        region_embeds = self.encode_regions(region_crops, full_image=full_image)

        text_norm = F.normalize(text_embeds, dim=-1)
        region_norm = F.normalize(region_embeds, dim=-1)
        similarity = torch.matmul(text_norm, region_norm.T)

        return tokens, similarity, region_embeds

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
