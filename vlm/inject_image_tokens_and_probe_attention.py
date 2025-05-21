import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import List, Tuple

@torch.no_grad()
def inject_image_tokens_and_probe_attention(
    model,
    processor,
    image: Image.Image,
    text: str,
    target_token: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:

    model.eval()
    tokenizer = processor.tokenizer

    # 1. Get image patch embeddings
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = image_inputs["pixel_values"]
    if pixel_values.ndim == 5:
        pixel_values = pixel_values[:, 0]  # remove time/frame dim
    with torch.no_grad():
        image_hidden = model.model.vision_model(pixel_values=pixel_values).last_hidden_state
        image_embeds = model.model.connector(image_hidden)  # project to decoder input space
        image_token_count = image_embeds.shape[1]

    # 2. Tokenize text and get embeddings
    text_inputs = tokenizer(text, return_tensors="pt").to(device)
    text_ids = text_inputs["input_ids"]
    text_tokens = tokenizer.convert_ids_to_tokens(text_ids[0])
    text_embeds = model.model.get_input_embeddings()(text_ids)  # (1, T, D)

    # 3. Concatenate [image_embeds] + [text_embeds]
    full_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (1, I+T, D)

    # 4. Attention mask and position IDs
    full_mask = torch.cat([
        torch.ones((1, image_token_count), dtype=torch.long, device=device),
        text_inputs["attention_mask"]
    ], dim=1)
    position_ids = torch.arange(full_embeds.shape[1], device=device).unsqueeze(0)

    # 5. Forward pass
    outputs = model.model(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        position_ids=position_ids,
        output_attentions=True,
        return_dict=True
    )

    # 6. Find target token index (within text_tokens)
    def normalize(tok): return tok.lstrip("Ġ▁_").lower()
    norm_target = normalize(target_token)
    tok_idx = next((i for i, tok in enumerate(text_tokens) if normalize(tok) == norm_target), None)
    if tok_idx is None:
        raise ValueError(f"Token '{target_token}' not found in: {text_tokens}")
    full_tok_idx = image_token_count + tok_idx

    # 7. Get attention from that token to all previous tokens
    attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    avg_attn = attn[:, full_tok_idx, :].mean(dim=0)  # (seq_len,)
    attn_to_image = avg_attn[:image_token_count]
    attn_to_text = avg_attn[image_token_count:]

    # 8. Fake token names for display
    image_tokens = [f"<img_{i:02d}>" for i in range(image_token_count)]
    all_tokens = image_tokens + text_tokens

    return all_tokens, attn_to_image.cpu(), attn_to_text.cpu(), avg_attn.cpu()
