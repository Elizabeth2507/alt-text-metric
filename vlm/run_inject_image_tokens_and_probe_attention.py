import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from inject_image_tokens_and_probe_attention import inject_image_tokens_and_probe_attention

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)


image = Image.open("data/school_bus.jpg").convert("RGB")
text = "A yellow school bus is parked on the road."

tokens, attn_img, attn_txt, attn_all = inject_image_tokens_and_probe_attention(
    model, processor, image, text, target_token="bus", device=device
)

print("Top attention to image patches:")
for i, score in enumerate(attn_img):
    print(f"{tokens[i]}: {score:.4f}")
