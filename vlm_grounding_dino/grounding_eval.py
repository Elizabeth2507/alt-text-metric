import torch
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel, pipeline

import sys
import os
# Add absolute path to modules/grounding_dino so "groundingdino.*" works
GROUNDING_DINO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modules", "grounding_dino"))
sys.path.insert(0, GROUNDING_DINO_ROOT)


from groundingdino.util.inference import load_model, predict

from dotenv import load_dotenv
from huggingface_hub import login

# Load variables from .env file
load_dotenv()

# Authenticate using token from .env
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in the .env file.")

# === CONFIG ===
CONFIG_PATH = "modules/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "vlm_grounding_dino/weights/groundingdino_swint_ogc.pth"

# === Load models globally ===
print("Loading models...")
dino_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")


# HuggingFace SRL pipeline (lightweight)
# from transformers import pipeline



# === SRL-based keyword extractor (Hugging Face) ===
def extract_srl_keywords(caption):
    results = srl_pipeline(caption)
    verbs = set()
    nouns = set()

    for item in results:
        label = item['entity_group']
        word = item['word'].lower()
        if label.startswith("V"):       # Verb
            verbs.add(word)
        elif "ARG" in label:            # Argument
            nouns.add(word)

    keywords = list(nouns | verbs)
    return keywords, list(nouns), list(verbs)

# === Grounding DINO predictor ===
def detect_with_grounding_dino(model, image_pil, keywords, box_thresh=0.3, text_thresh=0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image_pil,
        caption=", ".join(keywords),
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )
    return boxes, phrases

# === CLIP similarity ===
def match_with_clip(processor, model, image_pil, phrases, full_caption):
    inputs = processor(text=phrases + [full_caption], images=image_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    image_embeds = outputs.image_embeds  # (1, hidden)
    text_embeds = outputs.text_embeds    # (len(phrases)+1, hidden)
    sims = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    return sims

# === Visualization ===
def draw_boxes(image_path, boxes, phrases, output_path="output.jpg"):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    for box, phrase in zip(boxes, phrases):
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(image, phrase, (x0, max(y0 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(output_path, image)

# === Main pipeline ===
def run_pipeline(image_path, caption):
    image_pil = Image.open(image_path).convert("RGB")

    print("\n[1] Extracting SRL keywords...")
    all_keywords, obj_keywords, verb_keywords = extract_srl_keywords(caption)
    print(f"Objects: {obj_keywords}")
    print(f"Verbs:   {verb_keywords}")

    print("\n[2] Grounding with Grounding DINO...")
    boxes, phrases = detect_with_grounding_dino(dino_model, image_pil, all_keywords)

    if len(phrases) == 0:
        print("No objects/actions grounded.")
        return

    print("\n[3] Matching grounded phrases with full caption...")
    sims = match_with_clip(clip_processor, clip_model, image_pil, phrases, caption)

    print("\n--- Phrase Alignment Scores ---")
    for phrase, score in zip(phrases, sims):
        label = "VERB" if phrase in verb_keywords else "OBJ"
        print(f"{label:4s} | {phrase:30s} | similarity: {score.item():.4f}")

    print("\n[4] Drawing result image...")
    draw_boxes(image_path, boxes, phrases, output_path="output.jpg")
    print("Saved: output.jpg")

# === Example ===
if __name__ == "__main__":
    img_path = "data/school_bus.jpg"
    cap = "A yellow school bus is parked on the road."
    run_pipeline(img_path, cap)
