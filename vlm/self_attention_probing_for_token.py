import torch
from typing import List, Tuple
from PIL import Image


def self_attention_probing_for_token(
    model,
    processor,
    image: Image.Image,
    text: str,
    target_token: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts decoder self-attention for a token w.r.t. earlier image and text tokens.

    Args:
        model: HuggingFace AutoModelForImageTextToText (e.g., SmolVLM2)
        processor: Associated processor
        image: PIL image
        text: Generated or input caption
        target_token: Token string to inspect attention (e.g. "bus")
        device: Device

    Returns:
        full_tokens: list of tokens (image_tokens + text_tokens)
        attention_to_image: attention weights from target_token to image tokens
        attention_to_text: attention weights from target_token to text tokens
        avg_attention: full attention vector (image + text)
    """
    model.eval()

    # Tokenize text only
    text_inputs = processor.tokenizer(text, return_tensors="pt").to(device)
    text_tokens = processor.tokenizer.tokenize(text)

    # Process image input
    image_inputs_raw = processor(images=[image], return_tensors="pt")
    image_inputs = {
        k: v.to(device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in image_inputs_raw.items()
        if k == "pixel_values"
    }

    # Get image token count by forwarding through model.model.vision_model
    with torch.no_grad():
        pixel_values = image_inputs["pixel_values"]
        if pixel_values.ndim == 5:
            pixel_values = pixel_values[:, 0]  # Drop dummy time/frame dimension
        vision_outputs = model.model.vision_model(pixel_values=pixel_values)
        image_token_count = vision_outputs.last_hidden_state.shape[1]

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(
            **image_inputs,
            **text_inputs,
            output_attentions=True,
            return_dict=True
        )

    # Extract decoder self-attention from last layer
    # Shape: (num_heads, tgt_len, src_len)
    attentions = outputs.attentions[-1][0]

    # Build simulated full token list: image tokens + text tokens
    image_tokens = [f"<img_{i:02d}>" for i in range(image_token_count)]
    full_tokens = image_tokens + text_tokens

    # Normalize function
    def normalize(tok: str) -> str:
        return tok.lstrip("Ġ▁_").lower()

    normalized_target = normalize(target_token)
    token_offset = image_token_count
    token_idx = next(
        (i + token_offset for i, tok in enumerate(text_tokens) if normalize(tok) == normalized_target),
        None
    )
    if token_idx is None:
        raise ValueError(f"Token '{target_token}' not found in: {text_tokens}")

    # Get average attention from this token to all previous tokens
    attention_weights = attentions[:, token_idx, :]  # (num_heads, seq_len)
    avg_attention = attention_weights.mean(dim=0)    # (seq_len,)

    # Split
    attention_to_image = avg_attention[:image_token_count]
    attention_to_text = avg_attention[image_token_count:]

    return full_tokens, attention_to_image.cpu(), attention_to_text.cpu(), avg_attention.cpu()
