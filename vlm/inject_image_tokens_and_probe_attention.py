import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import List, Tuple
from PIL import ImageDraw


import os
os.makedirs("outputs", exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np


def save_original_patch_region_overlay(image: Image.Image, grid_size: int = 32, reduction_factor: int = 4):
    """
    Visualizes how the image was split into 32×32 patches and then grouped into 8×8 tokens.

    Each final token is represented as a colored box showing the region of the original 32×32 grid it covers.
    """
    from PIL import ImageDraw
    img = image.resize((grid_size * 32, grid_size * 32)).convert("RGBA")
    overlay = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(overlay)

    color = (0, 255, 0, 80)  # translucent green
    for token_idx in range((grid_size // reduction_factor) ** 2):
        region = get_original_patch_region(token_idx, reduction_factor, grid_size)
        (row_start, row_end), (col_start, col_end) = region["input_region"]

        x0 = col_start * 32
        y0 = row_start * 32
        x1 = col_end * 32
        y1 = row_end * 32

        draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
        draw.text((x0 + 2, y0 + 2), f"{token_idx:02d}", fill=(255, 255, 255, 255))

    out_img = Image.alpha_composite(img, overlay)
    out_img.save("outputs/original_patch_regions_overlay.png")
    print("Saved original 32×32→8×8 token region overlay to outputs/original_patch_regions_overlay.png")



def get_original_patch_region(token_idx, reduction_factor=4, input_grid=32):
    """
    Maps a reduced token index (e.g., from 8x8 grid) back to its original region in the input patch grid (e.g., 32x32).
    """
    grid_size = input_grid // reduction_factor  # e.g., 8
    row = token_idx // grid_size
    col = token_idx % grid_size
    return {
        "output_patch": (row, col),
        "input_region": (
            (row * reduction_factor, (row + 1) * reduction_factor),
            (col * reduction_factor, (col + 1) * reduction_factor),
        )
    }



def save_attention_overlay(
    image: Image.Image,
    attn_scores: torch.Tensor,
    target_token: str,
    grid_size: int = 32,
    draw_labels: bool = True
):
    """
    Draws attention overlay from a token to image patches and saves the result.

    Args:
        image (PIL.Image): The original image.
        attn_scores (torch.Tensor): Attention scores to image patches (length = grid_size²).
        target_token (str): The text token source of attention (for title and filename).
        grid_size (int): Grid resolution (e.g. 8 for 8x8).
        draw_labels (bool): Whether to write patch indices in the boxes.
    """
    import matplotlib.pyplot as plt
    from PIL import ImageDraw, ImageFont
    import numpy as np
    os.makedirs("outputs", exist_ok=True)

    attn = attn_scores.detach().cpu().numpy()
    patch_count = grid_size * grid_size
    assert len(attn) == patch_count, f"Expected {patch_count} patches but got {len(attn)}"

    # Normalize to [0, 1] for color intensity
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Resize image to match patch grid (scale up for better visibility)
    img = image.resize((grid_size * 32, grid_size * 32)).convert("RGBA")
    draw = ImageDraw.Draw(img)

    for i, score in enumerate(attn_norm):
        row = i // grid_size
        col = i % grid_size
        x0, y0 = col * 32, row * 32
        x1, y1 = x0 + 32, y0 + 32
        red_intensity = int(score * 255)

        # Draw translucent red rectangle with alpha blending
        overlay = Image.new("RGBA", img.size)
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, red_intensity))
        img = Image.alpha_composite(img, overlay)

        if draw_labels:
            # draw.text((x0 + 2, y0 + 2), str(i), fill=(255, 255, 255, 255))
            row, col = i // grid_size, i % grid_size
            draw.text((x0 + 2, y0 + 2), f"{row},{col}", fill=(255, 255, 255, 255))

    # Save result
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Attention to patches from '{target_token}'")
    plt.tight_layout()

    out_path = f"outputs/attn_overlay_{target_token}.png"
    plt.savefig(out_path)
    print(f" Saved overlay with patch IDs to {out_path}")
    plt.close()


def save_text_token_to_image_attention(token_list: List[str], attn: torch.Tensor, image_token_count: int, target_token: str):
    """Save heatmap of how each text token attends to image patches."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    text_start = image_token_count
    text_tokens = token_list[image_token_count:]
    token_count = len(text_tokens)

    attn_img = attn[:, text_start:text_start+token_count, :image_token_count].mean(dim=0)  # (T_text, N_img)

    plt.figure(figsize=(12, 0.4 * token_count))
    sns.heatmap(attn_img.detach().cpu().numpy(), cmap="YlOrRd", cbar=True, xticklabels=False)
    plt.yticks(np.arange(token_count) + 0.5, text_tokens, rotation=0)
    plt.xlabel("Image Patch Index")
    plt.title("Text tokens → Image patch attention (mean over heads)")
    plt.tight_layout()

    path = f"outputs/token_patch_heatmap_{target_token}.png"
    plt.savefig(path)
    print(f" Saved token→image heatmap to {path}")
    plt.close()


def save_per_head_patch_attention_plot(
    attn: torch.Tensor,
    full_tok_idx: int,
    image_token_count: int,
    target_token: str,
    save_path: str = None
):
    """
    Save per-head attention plot from a given text token to image patches.
    
    Args:
        attn (torch.Tensor): Attention tensor of shape (num_heads, seq_len, seq_len)
        full_tok_idx (int): Index of the target token in the full sequence
        image_token_count (int): Number of image tokens at the beginning of the sequence
        target_token (str): The token used for the attention probe (e.g., "bus")
        save_path (str): Optional; path to save the plot. If None, uses default in outputs/
    """
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs("outputs", exist_ok=True)

    # Extract attention from each head to the image region
    head_attn = attn[:, full_tok_idx, :image_token_count]  # (num_heads, image_tokens)

    plt.figure(figsize=(10, 6))
    for i, vec in enumerate(head_attn):
        plt.plot(vec.detach().cpu().numpy(), label=f'head {i}')
    
    plt.title(f'Attention from token \"{target_token}\" to image patches (per head)')
    plt.xlabel("Image Patch Index")
    plt.ylabel("Attention Score")
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = f"outputs/per_head_attention_{target_token}.png"

    plt.savefig(save_path)
    print(f" Saved per-head attention plot to {save_path}")
    plt.close()


def get_top_heads(attn: torch.Tensor, token_idx: int, image_token_count: int, top_k: int = 3) -> List[int]:
    """
    Returns indices of top-k heads that give the most attention from a given token to image patches.
    
    Args:
        attn (torch.Tensor): Attention tensor of shape (num_heads, seq_len, seq_len)
        token_idx (int): Index of the target token in the sequence
        image_token_count (int): Number of image tokens at the start of the sequence
        top_k (int): Number of top heads to return

    Returns:
        List[int]: Indices of the top-k attention heads
    """
    with torch.no_grad():
        token_to_image = attn[:, token_idx, :image_token_count]  # shape: (num_heads, image_tokens)
        total_per_head = token_to_image.sum(dim=1)  # shape: (num_heads,)
        top_heads = torch.topk(total_per_head, k=top_k).indices.tolist()
    return top_heads


def save_top_head_attention_overlay(
    image: Image.Image,
    attn: torch.Tensor,
    token_idx: int,
    image_token_count: int,
    target_token: str,
    grid_size: int = 32,
    top_heads: List[int] = None
):
    """
    Overlays average attention to image patches from top attention heads onto the image.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs("outputs", exist_ok=True)

    if top_heads is None:
        raise ValueError("You must pass top_heads as a list of attention head indices.")

    # Extract and average the attention from the top heads
    patch_attn = attn[top_heads, token_idx, :image_token_count]  # (top_k, I)
    avg_attn = patch_attn.mean(dim=0).detach().cpu().numpy()  # (I,)

    # Normalize for display
    attn_norm = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)

    # Draw overlay
    img = image.resize((grid_size * 32, grid_size * 32)).convert("RGBA")
    overlay = Image.new("RGBA", img.size)
    overlay_draw = ImageDraw.Draw(overlay)

    for i, score in enumerate(attn_norm):
        row = i // grid_size
        col = i % grid_size
        x0, y0 = col * 32, row * 32
        x1, y1 = x0 + 32, y0 + 32
        red_intensity = int(score * 255)
        overlay_draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, red_intensity))

    img = Image.alpha_composite(img, overlay)

    # Save result
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Top-{len(top_heads)} head attention from '{target_token}'")
    plt.tight_layout()
    out_path = f"outputs/attn_overlay_top_heads_{target_token}.png"
    plt.savefig(out_path)
    print(f" Saved top-head overlay to {out_path}")
    plt.close()



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
    # image_inputs = processor(images=[image], return_tensors="pt").to(device)
    # image_inputs = {
    #     k: v.to(device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
    #     for k, v in image_inputs.items()
    # }
    image_inputs = processor(images=[image], return_tensors="pt").to(device)
    pixel_values = image_inputs["pixel_values"]
    if pixel_values.ndim == 5:
        pixel_values = pixel_values[:, 0]  # remove time/frame dim
    with torch.no_grad():
        image_hidden = model.model.vision_model(pixel_values=pixel_values).last_hidden_state
        # print("image_hidden.shape:", image_hidden.shape)  # (B, N_patches, D)
        # --- Track 2D patch layout ---
        patch_count = image_hidden.shape[1]  # 1024
        grid_size = int(patch_count ** 0.5)  # Should be 32
        assert grid_size * grid_size == patch_count, "Patch count must be a perfect square"

        patch_coords = [(i // grid_size, i % grid_size) for i in range(patch_count)]
        print("First 10 patch coordinates (row, col):", patch_coords[:10])

        image_embeds = model.model.connector(image_hidden)  # project to decoder input space
        print("image_hidden.shape:", image_hidden.shape)   # (1, 1024, 768)
        print("image_embeds.shape:", image_embeds.shape)   # (1, 64, 960)

        # 👇 INSERT HERE
        print("Patch token shape:", image_embeds.shape)
        print("First 5 patch tokens:\n", image_embeds[0, :5])  # [num_patches, dim]
        image_token_count = image_embeds.shape[1]

    # 2. Tokenize text and get embeddings
    text_inputs = tokenizer(text, return_tensors="pt").to(device)
    text_ids = text_inputs["input_ids"]
    text_tokens = tokenizer.convert_ids_to_tokens(text_ids[0])
    text_embeds = model.model.get_input_embeddings()(text_ids)  # (1, T, D)

    # Tokenize alt-text (i.e., caption) to evaluate
    desc_inputs = tokenizer(text, return_tensors="pt").to(device)
    desc_ids = desc_inputs["input_ids"]
    desc_tokens = tokenizer.convert_ids_to_tokens(desc_ids[0])
    desc_embeds = model.model.get_input_embeddings()(desc_ids)

    # 3. Concatenate [image_embeds] + [text_embeds]
    full_embeds = torch.cat([image_embeds, desc_embeds], dim=1)  # (1, I+T, D)

    # 4. Attention mask and position IDs
    seq_len = full_embeds.shape[1]
    full_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 5. Forward pass
    try:
        outputs = model.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            output_attentions=True,
            return_dict=True
        )
    except ValueError:
        """
            You must explicitly pass input_ids in addition to inputs_embeds, even if you are not using them directly.
            This satisfies the model’s internal checks and prevents:
            ValueError: When first calling the model, if input_embeds are passed, input_ids should not be None.
            
            Even though the input_ids don’t match the full_embeds length (because you concatenated image + text), 
            this satisfies the internal requirement — the model won't actually use input_ids when inputs_embeds is supplied.
            This quirk is common in some Hugging Face models with generation or multitask wrappers.
        """
        outputs = model.model(
            # input_ids=text_ids,  # fallback fix
            # # Dummy token to activate decoder (neutral, not semantic)
            input_ids = tokenizer("", return_tensors="pt").input_ids.to(device),
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            output_attentions=True,
            return_dict=True
        )

    # --- DEBUG: Sanity checks ---
    print(f"\n image_embeds.shape = {image_embeds.shape}")  # (1, N_img, D)
    print(f" text_embeds.shape = {text_embeds.shape}")      # (1, N_text, D)
    print(f" full_embeds.shape = {full_embeds.shape}")      # (1, N_img + N_text, D) — total tokens
    print(f" Number of image tokens: {image_token_count}")
    print(f" Sample tokens (first 10): {[t for t in text_tokens[:10]]}")


    # 6. Find target token index (within text_tokens)
    def normalize(tok): return tok.lstrip("Ġ▁_").lower()
    norm_target = normalize(target_token)
    tok_idx = next((i for i, tok in enumerate(text_tokens) if normalize(tok) == norm_target), None)
    if tok_idx is None:
        raise ValueError(f"Token '{target_token}' not found in: {text_tokens}")
    full_tok_idx = image_token_count + tok_idx

    # 7. Get attention from that token to all previous tokens
    attn = outputs.attentions[-1][0]  # shape: (num_heads, seq_len, seq_len)

    save_per_head_patch_attention_plot(attn, full_tok_idx, image_token_count, target_token)

    avg_attn = attn[:, full_tok_idx, :].mean(dim=0)  # shape: (seq_len,)
    attn_to_image = avg_attn[:image_token_count]


    attn_to_text = avg_attn[image_token_count:]

    # 8. Build token names
    image_tokens = [f"<img_{i:02d}>" for i in range(image_token_count)]
    all_tokens = image_tokens + text_tokens

    # --- Save attention visualizations ---
    save_attention_overlay(image, attn_to_image, target_token=target_token, grid_size=8)
    save_text_token_to_image_attention(all_tokens, attn, image_token_count, target_token=target_token)

    # Extract top heads
    top_heads = get_top_heads(attn, full_tok_idx, image_token_count, top_k=3)

    # Save overlay using only those heads
    save_top_head_attention_overlay(
        image=image,
        attn=attn,
        token_idx=full_tok_idx,
        image_token_count=image_token_count,
        target_token=target_token,
        grid_size=8,
        top_heads=top_heads
    )

    save_original_patch_region_overlay(image, grid_size=32, reduction_factor=4)

    return all_tokens, attn_to_image.cpu(), attn_to_text.cpu(), avg_attn.cpu(), patch_coords
