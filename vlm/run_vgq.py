from vgq import SmolVLMEmbedder, VGQScorer 
from utils import load_region_crops, load_full_image

def main():
    # Step 1: Load image crops and full image
    region_crops = load_region_crops("region_outputs")     # List[Image.Image]
    full_image = load_full_image("data/school_bus.jpg")     # PIL.Image.Image
    alt_text = "A yellow school bus is parked on the road."

    # Step 2a: Create embedder
    embedder = SmolVLMEmbedder()

    # ================================
    # 🔹 USAGE 1: Region-only embeddings (no CLS)
    # ================================
    print("\n=== Matching using REGION-ONLY embeddings ===")
    scorer_basic = VGQScorer(embedder, use_cls_context=False)

    # Get region embeddings only
    region_embeds_pure = scorer_basic.encode_regions(region_crops)

    # Token-to-region matching (cosine)
    matches_basic = scorer_basic.get_token_region_matches(
        text=alt_text,
        region_crops=region_crops
    )

    for match in matches_basic:
        print(f"{match['token']:>12} → Region {match['region_id']} (score={match['score']:.3f})  Match? {match['matched']}")

    # ================================
    # 🔹 USAGE 2: Region + [CLS] context from full image
    # ================================
    print("\n=== Matching using REGION + CLS GLOBAL CONTEXT ===")
    scorer_with_cls = VGQScorer(embedder, use_cls_context=True)

    # Get fused embeddings
    region_embeds_with_cls = scorer_with_cls.encode_regions(region_crops, full_image=full_image)

    # Token-to-region matching with CLS (cosine)
    matches_with_cls = scorer_with_cls.get_token_region_matches(
        text=alt_text,
        region_crops=region_crops,
        full_image=full_image  # Required here
    )

    for match in matches_with_cls:
        print(f"{match['token']:>12} → Region {match['region_id']} (score={match['score']:.3f})  Match? {match['matched']}")


if __name__ == "__main__":
    main()
