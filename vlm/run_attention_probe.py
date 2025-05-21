import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from attention_probe import self_attention_probing_for_token



device = "cpu"
model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Prepare inputs
image = Image.open("data/school_bus.jpg").convert("RGB")
text = "A yellow school bus is parked on the road."

tokens, attention_weights = self_attention_probing_for_token(
    model=model,
    processor=processor,
    image=image,
    text=text,
    target_token="bus",
    device=device
)

# Visualize or print results
print("Tokens:", tokens)
print("Attention weights for token 'bus':")
for i, (tok, weight) in enumerate(zip(tokens, attention_weights.tolist())):
    print(f"{i:2d}: {tok:15s} -> {weight:.4f}")
