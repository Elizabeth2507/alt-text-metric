# vgq_decoder_probe.py
import torch

def get_decoder_attention(model, processor, image, text, device=None, layer_idx=-1, head_idx=None):
    """
    Extract decoder cross-attention maps for each token.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # === STEP 1: Process image
    image_inputs = processor(images=[image], return_tensors="pt")
    image_inputs = {
        k: v.to(device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in image_inputs.items()
    }

    # === STEP 2: Tokenize text
    text_inputs = processor.tokenizer(text, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # === STEP 3: Merge
    inputs = {
        "pixel_values": image_inputs["pixel_values"],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"]
    }

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    # Get tokens and cross-attention maps
    tokens = processor.tokenizer.tokenize(text)
    cross_attn = outputs.cross_attentions  # list of tensors [layers], shape: (1, heads, tokens, patches)

    # Select layer and optionally average over heads
    selected = cross_attn[layer_idx].squeeze(0)  # (heads, tokens, patches)
    attn = selected.mean(0) if head_idx is None else selected[head_idx]  # (tokens, patches)

    return tokens, attn.cpu()
