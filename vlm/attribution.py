import torch
import torch.nn.functional as F
from typing import List, Tuple
from PIL import Image


def grad_attribution_for_token(
    model,
    processor,
    image: Image.Image,
    text: str,
    target_token: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List[str], torch.Tensor]:
    """
    Grad × Input attribution for a token w.r.t. image patches (SmolVLM2-style model).

    Args:
        model: HuggingFace AutoModelForImageTextToText
        processor: Corresponding AutoProcessor
        image: PIL image
        text: input caption
        target_token: token string to attribute (e.g., "parked")
        device: "cuda" or "cpu"

    Returns:
        tokens: list of token strings
        attribution_map: normalized (H, W) tensor
    """
    model.eval()

    # Process image
    image_inputs = processor(images=[image], return_tensors="pt")
    image_inputs = {
        k: v.to(device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in image_inputs.items()
    }

    # Tokenize text
    text_inputs = processor.tokenizer(text, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    tokens = processor.tokenizer.tokenize(text)

    # Enable gradient tracking on image
    pixel_values = image_inputs["pixel_values"].clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(
        pixel_values=pixel_values,
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        return_dict=True
    )
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Normalize function for token matching
    def normalize(token: str) -> str:
        return token.lstrip("Ġ▁_").lower()

    normalized_target = target_token.lower()
    token_idx = next(
        (i for i, tok in enumerate(tokens) if normalize(tok) == normalized_target),
        None
    )
    if token_idx is None:
        raise ValueError(f"Token '{target_token}' not found in: {tokens}")

    # Select target logit and compute backward
    target_logits = logits[0, token_idx]
    token_id = text_inputs["input_ids"][0, token_idx]
    loss = target_logits[token_id]
    loss.backward()

    grads = pixel_values.grad  # (1, C, H, W)

    # Grad × Input attribution
    # attribution = (pixel_values * grads).sum(dim=1).squeeze(0)  # (H, W)
    # Grad × Input, reduce over channels (C) and time (T)
    # pixel_values: (1, T, C, H, W)
    # grads:        (1, T, C, H, W)

    # Grad × Input → sum over channels (C) and time (T)
    attribution = (pixel_values * grads).sum(dim=2)  # (1, T, H, W)
    # attribution = (pixel_values * grads).sum(dim=2).mean(dim=1).squeeze(0)  # (H, W)
    attribution = attribution.sum(dim=1)             # (1, H, W)
    attribution = attribution.squeeze(0)             # (H, W)
    
    print("Attribution stats:", attribution.min().item(), attribution.max().item(), attribution.mean().item())

    attribution = F.relu(attribution)
    attribution = attribution / (attribution.max() + 1e-8)

    return tokens, attribution.detach().cpu()
