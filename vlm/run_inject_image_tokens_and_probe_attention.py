import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from inject_image_tokens_and_probe_attention import inject_image_tokens_and_probe_attention, get_original_patch_region

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)


image = Image.open("data/1115.jpg").convert("RGB")
text = "The car."

tokens, attn_img, attn_txt, attn_all, patch_coords = inject_image_tokens_and_probe_attention(
    model, processor, image, text, target_token="the", device=device
)

# print("\n Mapping each reduced image token to its region in the original 32x32 patch grid:")

# for i in range(len(patch_coords)):  # or: range(64)
#     region_info = get_original_patch_region(i, reduction_factor=4, input_grid=32)
#     out_patch = region_info["output_patch"]
#     in_region = region_info["input_region"]
#     print(f"Image token {i:02d} → Output patch {out_patch} → Covers input rows {in_region[0]}, cols {in_region[1]}")


print("Top attention to image patches:")
for i, score in enumerate(attn_img):
    print(f"{tokens[i]}: {score:.4f}")

# Number of image patch tokens
image_token_count = len(patch_coords)

# Print attention from text token to each image patch (with spatial info)
print("Top attention to image patches:")
for i, (score, (row, col)) in enumerate(zip(attn_img, patch_coords)):
    print(f"Image token {i:03d} → Patch ({row:02d}, {col:02d}) → Attention: {score:.4f}")
