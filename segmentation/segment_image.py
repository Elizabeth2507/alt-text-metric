import numpy as np
import cv2

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import (
    EfficientViTSamAutomaticMaskGenerator,
    EfficientViTSamPredictor
)
from efficientvit.models.utils import build_kwargs_from_config


def load_sam_model(model_name="efficientvit_sam_l2", pretrained=True):
    model = create_efficientvit_sam_model(model_name, pretrained=pretrained)
    model.eval()
    return model


def run_sam_on_image(model, image_np: np.ndarray, pred_iou_thresh=0.8, stability_score_thresh=0.85, min_mask_region_area=100):
    """Automatic segmentation with no prompts"""
    mask_generator = EfficientViTSamAutomaticMaskGenerator(
        model,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        **build_kwargs_from_config({}, EfficientViTSamAutomaticMaskGenerator),
    )
    masks = mask_generator.generate(image_np)
    return [m["segmentation"] for m in masks]


def run_sam_with_points(model, image_np: np.ndarray, points, labels, multimask=False):
    """Point-guided segmentation (e.g., [(x,y)], [1])"""
    predictor = EfficientViTSamPredictor(model)
    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=multimask
    )
    return masks


def run_sam_with_box(model, image_np: np.ndarray, box, multimask=False):
    """Box-guided segmentation (e.g., [[x0, y0, x1, y1]])"""
    predictor = EfficientViTSamPredictor(model)
    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(box),
        multimask_output=multimask
    )
    return masks


def extract_region_crops(image_np, masks, min_area=1000):
    """Extract rectangular crops from binary masks."""
    region_crops = []
    for mask in masks:
        if np.sum(mask) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        region_crops.append(image_np[y:y + h, x:x + w])
    return region_crops
