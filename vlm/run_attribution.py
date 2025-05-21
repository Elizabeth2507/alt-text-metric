import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from attribution import grad_attribution_for_token
from visualize import show_attribution_overlay


device = "cpu"
model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Prepare inputs
image = Image.open("data/school_bus.jpg").convert("RGB")
text = "A yellow school bus is parked on the road."

# Run Grad × Input attribution for a token
tokens, attribution = grad_attribution_for_token(
    model=model,
    processor=processor,
    image=image,
    text=text,
    target_token="bus",
    device=device
)

# Optional: print attribution summary
print("Tokens:", tokens)
print("Attribution shape:", attribution.shape)

show_attribution_overlay(
    image=image,
    attribution=attribution,
    alpha=0.5,
    save_path="outputs/attribution_parked.png",
    show=False  # change to True if you want to view interactively
)

