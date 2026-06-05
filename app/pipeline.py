from __future__ import annotations

from .features import analyze_edges_and_corners, extract_centering_metrics
from .grading import estimate_grade
from .surface_ai import analyze_surface
from .vision_preprocess import decode_image, preprocess_card_image


def analyze_card_image(image_bytes: bytes, filename: str | None = None) -> dict:
    image = decode_image(image_bytes)
    preprocess_result = preprocess_card_image(image)

    centering = extract_centering_metrics(preprocess_result.card_image)
    edges_and_corners = analyze_edges_and_corners(preprocess_result.card_image)
    surface = analyze_surface(preprocess_result.card_image)
    grading = estimate_grade(centering, edges_and_corners, surface)

    return {
        "filename": filename or "uploaded_image",
        "preprocessing": {
            "original_size": {
                "width": preprocess_result.original_size[0],
                "height": preprocess_result.original_size[1],
            },
            "resized_size": {
                "width": preprocess_result.resized_size[0],
                "height": preprocess_result.resized_size[1],
            },
            "card_size": {
                "width": preprocess_result.card_size[0],
                "height": preprocess_result.card_size[1],
            },
            "card_detected": preprocess_result.card_detected,
            "contour_area_ratio": preprocess_result.contour_area_ratio,
        },
        "features": {
            "centering": centering,
            "corners_and_edges": edges_and_corners,
        },
        "surface": surface,
        "grading": grading,
        "summary": {
            "centering_lr": centering["left_right_ratio"],
            "centering_tb": centering["top_bottom_ratio"],
            "edge_wear_level": edges_and_corners["wear_level"],
            "surface_damage_score": surface["damage_score"],
            "estimated_grade_range": grading["estimated_range"],
        },
    }

