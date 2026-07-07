from __future__ import annotations

import cv2
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _find_peak_index(profile: np.ndarray, start: int, end: int) -> int:
    start = max(start, 0)
    end = min(end, profile.shape[0])
    if end - start <= 3:
        return (start + end) // 2
    segment = profile[start:end]
    return int(start + int(np.argmax(segment)))


def extract_centering_metrics(card_image: np.ndarray) -> dict:
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    grad_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))

    vertical_profile = grad_x.mean(axis=0)
    horizontal_profile = grad_y.mean(axis=1)

    # Smooth the entire profiles first to avoid boundary shifts near segment edges
    window_x = max(3, int(vertical_profile.shape[0] * 0.01))
    kernel_x = np.ones(window_x, dtype=np.float32) / float(window_x)
    smoothed_x = np.convolve(vertical_profile, kernel_x, mode="same")

    window_y = max(3, int(horizontal_profile.shape[0] * 0.01))
    kernel_y = np.ones(window_y, dtype=np.float32) / float(window_y)
    smoothed_y = np.convolve(horizontal_profile, kernel_y, mode="same")

    height, width = gray.shape

    left_transition = _find_peak_index(smoothed_x, int(width * 0.02), int(width * 0.18))
    right_transition = _find_peak_index(smoothed_x, int(width * 0.82), int(width * 0.98))
    top_transition = _find_peak_index(smoothed_y, int(height * 0.02), int(height * 0.18))
    bottom_transition = _find_peak_index(smoothed_y, int(height * 0.82), int(height * 0.98))

    # Get peak values on raw profiles to validate they are real transition edges
    left_peak_val = vertical_profile[left_transition]
    right_peak_val = vertical_profile[right_transition]
    top_peak_val = horizontal_profile[top_transition]
    bottom_peak_val = horizontal_profile[bottom_transition]

    # Validate transitions (minimum gradient magnitude + coordinate check) and fallback if invalid
    MIN_GRADIENT_THRESHOLD = 1.0
    if (right_transition <= left_transition or 
        left_peak_val < MIN_GRADIENT_THRESHOLD or 
        right_peak_val < MIN_GRADIENT_THRESHOLD):
        left_transition = int(width * 0.06)
        right_transition = int(width * 0.94)

    if (bottom_transition <= top_transition or 
        top_peak_val < MIN_GRADIENT_THRESHOLD or 
        bottom_peak_val < MIN_GRADIENT_THRESHOLD):
        top_transition = int(height * 0.06)
        bottom_transition = int(height * 0.94)

    left_border = max(1, left_transition)
    right_border = max(1, width - right_transition)
    top_border = max(1, top_transition)
    bottom_border = max(1, height - bottom_transition)

    lr_total = left_border + right_border
    tb_total = top_border + bottom_border

    left_pct = round((left_border / lr_total) * 100, 1)
    right_pct = round(100 - left_pct, 1)
    top_pct = round((top_border / tb_total) * 100, 1)
    bottom_pct = round(100 - top_pct, 1)

    lr_imbalance = abs(left_pct - right_pct)
    tb_imbalance = abs(top_pct - bottom_pct)
    max_imbalance = max(lr_imbalance, tb_imbalance)

    centering_score = _clamp(10.0 - (max_imbalance / 3.5), 1.0, 10.0)

    if max_imbalance <= 4:
        quality = "elite"
    elif max_imbalance <= 10:
        quality = "strong"
    elif max_imbalance <= 16:
        quality = "acceptable"
    else:
        quality = "off-center"

    return {
        "left_border_px": int(left_border),
        "right_border_px": int(right_border),
        "top_border_px": int(top_border),
        "bottom_border_px": int(bottom_border),
        "left_right_ratio": f"{int(round(left_pct))}/{int(round(right_pct))}",
        "top_bottom_ratio": f"{int(round(top_pct))}/{int(round(bottom_pct))}",
        "left_percent": left_pct,
        "right_percent": right_pct,
        "top_percent": top_pct,
        "bottom_percent": bottom_pct,
        "max_imbalance_percent": round(max_imbalance, 2),
        "centering_score": round(centering_score, 2),
        "centering_quality": quality,
    }


def _wear_level(score: float) -> str:
    if score < 0.2:
        return "minimal"
    if score < 0.4:
        return "light"
    if score < 0.65:
        return "moderate"
    return "heavy"


def analyze_edges_and_corners(card_image: np.ndarray) -> dict:
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    height, width = gray.shape

    border_band = max(6, int(min(height, width) * 0.04))
    border_mask = np.zeros_like(gray, dtype=np.uint8)
    border_mask[:border_band, :] = 1
    border_mask[-border_band:, :] = 1
    border_mask[:, :border_band] = 1
    border_mask[:, -border_band:] = 1

    border_pixels = max(1, int(np.count_nonzero(border_mask)))
    border_edge_density = float(np.count_nonzero(edges[border_mask == 1])) / float(border_pixels)
    border_std = float(np.std(gray[border_mask == 1]))

    edge_density_component = min(1.0, border_edge_density / 0.14)
    border_std_component = min(1.0, border_std / 60.0)
    edge_wear_score = _clamp(0.7 * edge_density_component + 0.3 * border_std_component, 0.0, 1.0)

    patch_size = max(16, int(min(height, width) * 0.12))
    corner_regions = [
        gray[:patch_size, :patch_size],
        gray[:patch_size, -patch_size:],
        gray[-patch_size:, :patch_size],
        gray[-patch_size:, -patch_size:],
    ]

    corner_scores: list[float] = []
    for patch in corner_regions:
        patch_edges = cv2.Canny(patch, 80, 170)
        patch_edge_density = float(np.count_nonzero(patch_edges)) / float(patch_edges.size)
        patch_std = float(np.std(patch))

        patch_score = _clamp(
            0.65 * min(1.0, patch_edge_density / 0.18) + 0.35 * min(1.0, patch_std / 65.0),
            0.0,
            1.0,
        )
        corner_scores.append(patch_score)

    corner_wear_score = float(np.mean(corner_scores))
    combined_wear_score = _clamp(0.55 * edge_wear_score + 0.45 * corner_wear_score, 0.0, 1.0)

    wear_level = _wear_level(combined_wear_score)
    if wear_level == "minimal":
        notes = ["Edges and corners appear clean with limited roughness."]
    elif wear_level == "light":
        notes = ["Minor wear patterns detected along edges or corner tips."]
    elif wear_level == "moderate":
        notes = ["Visible wear likely present on edges/corners, which can cap grade."]
    else:
        notes = ["Strong edge/corner roughness detected and likely grade limiting."]

    return {
        "edge_wear_score": round(edge_wear_score, 3),
        "corner_wear_score": round(corner_wear_score, 3),
        "combined_wear_score": round(combined_wear_score, 3),
        "wear_level": wear_level,
        "notes": notes,
    }

