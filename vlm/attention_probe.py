import torch
from typing import List, Tuple


def self_attention_probing_for_token(
    model,
    processor,
    image,
    text: str,
    target_token: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List[str], torch.Tensor]:
    """
    Extracts decoder self-attention for a token w.r.t. earlier image tokens.

    Args:
        model: HuggingFace AutoModelForImageTextToText (e.g., SmolVLM2)
        processor: Associated processor
        image: PIL Image
        text: Generated or input caption
        target_token: Token string to inspect attention (e.g. "bus")
        device: Device

    Returns:
        tokens: list of all tokens
        attention_map: attention weights from target_token to earlier tokens (1D tensor)
    """
    model.eval()

    # Tokenize
    text_inputs = processor.tokenizer(text, return_tensors="pt").to(device)
    tokens = processor.tokenizer.tokenize(text)

    # Process image
    # Process image
    image_inputs_raw = processor(images=[image], return_tensors="pt")
    # Filter only arguments that model expects (e.g. "pixel_values")
    allowed_keys = {"pixel_values"}
    image_inputs = {
        k: v.to(device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in image_inputs_raw.items()
        if k in allowed_keys
    }


    # Forward with attention outputs
    with torch.no_grad():
        outputs = model(
            **image_inputs,
            **text_inputs,
            output_attentions=True,
            return_dict=True
        )

    # Find target token index
    def normalize(token: str) -> str:
        return token.lstrip("Ġ▁_").lower()

    normalized_target = target_token.lower()
    token_idx = next(
        (i for i, tok in enumerate(tokens) if normalize(tok) == normalized_target),
        None
    )
    if token_idx is None:
        raise ValueError(f"Token '{target_token}' not found in: {tokens}")

    # Collect last layer decoder self-attention
    # SmolVLM2: causal decoder, attention shape = (batch, num_heads, tgt_len, src_len)
    attentions = outputs.attentions[-1][0]  # shape: (num_heads, tgt_len, src_len)

    # Get average attention from target_token across all heads
    attention_weights = attentions[:, token_idx, :]  # (num_heads, src_len)
    avg_attention = attention_weights.mean(dim=0)    # (src_len,)

    return tokens, avg_attention.detach().cpu()
