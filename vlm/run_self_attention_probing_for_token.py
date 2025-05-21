import torch
from self_attention_probing_for_token import self_attention_probing_for_token
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("data/school_bus.jpg").convert("RGB")
text = "A yellow school bus is parked on the road."

tokens, attn_img, attn_txt, attn_all = self_attention_probing_for_token(
    model=model,
    processor=processor,
    image=image,
    text=text,
    target_token="bus",
    device=device
)

print("Tokens:", tokens)
print("Attention to image sum:", attn_img.sum().item())
print("Attention to text sum :", attn_txt.sum().item())
