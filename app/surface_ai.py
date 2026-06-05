from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
from huggingface_hub import InferenceClient


SURFACE_LABELS = [
    "a clean trading card surface",
    "a trading card with light surface scratches",
    "a trading card with heavy surface scratches",
    "a trading card with discoloration spots",
    "a trading card with visible image noise",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _encode_jpeg(card_image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", card_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Unable to encode card image for surface model request.")
    return encoded.tobytes()


def _prediction_map(predictions: list[Any]) -> dict[str, float]:
    score_map: dict[str, float] = {}
    for item in predictions:
        if isinstance(item, dict):
            label = str(item.get("label", ""))
            score = float(item.get("score", 0.0))
        else:
            label = str(getattr(item, "label", ""))
            score = float(getattr(item, "score", 0.0))

        if label:
            score_map[label] = score
    return score_map


def _heuristic_surface_assessment(card_image: np.ndarray) -> dict:
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    scratch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright_lines = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, scratch_kernel)
    dark_lines = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, scratch_kernel)
    scratch_signal = (float(np.mean(bright_lines)) + float(np.mean(dark_lines))) / 255.0
    scratch_probability = _clamp(scratch_signal * 4.0)

    lab = cv2.cvtColor(card_image, cv2.COLOR_BGR2LAB)
    a_std = float(np.std(lab[:, :, 1]))
    b_std = float(np.std(lab[:, :, 2]))
    discolor_probability = _clamp((a_std + b_std) / 70.0)

    laplacian_variance = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    noise_probability = _clamp(laplacian_variance / 1500.0)

    damage_score = _clamp(
        0.6 * scratch_probability + 0.25 * discolor_probability + 0.15 * noise_probability
    )
    clean_probability = _clamp(1.0 - damage_score * 0.9)

    findings: list[str] = []
    if scratch_probability >= 0.4:
        findings.append("Potential scratch patterns detected.")
    if discolor_probability >= 0.45:
        findings.append("Possible color inconsistency/discoloration detected.")
    if noise_probability >= 0.45:
        findings.append("Surface texture appears noisy.")
    if not findings:
        findings.append("No strong surface damage signals detected by heuristic analysis.")

    return {
        "scratch_probability": round(scratch_probability, 3),
        "discoloration_probability": round(discolor_probability, 3),
        "noise_probability": round(noise_probability, 3),
        "clean_probability": round(clean_probability, 3),
        "damage_score": round(damage_score, 3),
        "findings": findings,
        "top_predictions": [],
    }


def _ai_surface_assessment(predictions: list[Any]) -> dict:
    scores = _prediction_map(predictions)

    clean = scores.get("a clean trading card surface", 0.0)
    light_scratches = scores.get("a trading card with light surface scratches", 0.0)
    heavy_scratches = scores.get("a trading card with heavy surface scratches", 0.0)
    discoloration = scores.get("a trading card with discoloration spots", 0.0)
    noise = scores.get("a trading card with visible image noise", 0.0)

    scratch_probability = _clamp(heavy_scratches + (0.65 * light_scratches))
    damage_score = _clamp(0.62 * scratch_probability + 0.25 * discoloration + 0.13 * noise)

    if clean > 0.45:
        damage_score = _clamp(damage_score * 0.8)
    elif clean < 0.2:
        damage_score = _clamp(damage_score + 0.08)

    findings: list[str] = []
    if scratch_probability >= 0.4:
        findings.append("AI model detected probable scratching.")
    if discoloration >= 0.35:
        findings.append("AI model detected potential discoloration.")
    if noise >= 0.35:
        findings.append("AI model detected elevated surface noise.")
    if damage_score < 0.2:
        findings.append("Surface appears mostly clean from AI perspective.")

    top_predictions = sorted(
        [{"label": label, "score": round(score, 3)} for label, score in scores.items()],
        key=lambda item: item["score"],
        reverse=True,
    )[:3]

    return {
        "scratch_probability": round(scratch_probability, 3),
        "discoloration_probability": round(discoloration, 3),
        "noise_probability": round(noise, 3),
        "clean_probability": round(clean, 3),
        "damage_score": round(damage_score, 3),
        "findings": findings,
        "top_predictions": top_predictions,
    }


def analyze_surface(card_image: np.ndarray) -> dict:
    heuristic_result = _heuristic_surface_assessment(card_image)

    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_VISION_MODEL", "openai/clip-vit-base-patch32")
    provider = os.getenv("HF_PROVIDER", "hf-inference")

    if not token:
        heuristic_result["source"] = "heuristic"
        heuristic_result["model"] = None
        heuristic_result["provider"] = None
        heuristic_result["api_error"] = "HF_TOKEN not set. Using heuristic-only surface detection."
        return heuristic_result

    try:
        client = InferenceClient(api_key=token, provider=provider)
        predictions = client.zero_shot_image_classification(
            image=_encode_jpeg(card_image),
            model=model,
            candidate_labels=SURFACE_LABELS,
        )
        ai_result = _ai_surface_assessment(predictions)
        ai_result["source"] = "huggingface_api"
        ai_result["model"] = model
        ai_result["provider"] = provider
        ai_result["api_error"] = None
        return ai_result
    except Exception as exc:  # pylint: disable=broad-except
        heuristic_result["source"] = "heuristic_fallback"
        heuristic_result["model"] = model
        heuristic_result["provider"] = provider
        heuristic_result["api_error"] = str(exc)
        return heuristic_result

