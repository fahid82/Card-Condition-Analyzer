from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _centering_base_score(centering: dict) -> tuple[float, str]:
    max_imbalance = float(centering.get("max_imbalance_percent", 0.0))

    if max_imbalance <= 2:
        return 10.0, "Perfect or near-perfect centering."
    if max_imbalance <= 5:
        return 9.0, "Slight centering variance."
    if max_imbalance <= 10:
        return 8.5, "Noticeable but acceptable centering offset."
    if max_imbalance <= 15:
        return 8.0, "Moderately off-center borders."
    if max_imbalance <= 20:
        return 7.0, "Clearly off-center borders."
    return 6.0, "Strong centering imbalance."


def _edge_wear_cap(wear_score: float) -> tuple[float, str] | tuple[None, None]:
    if wear_score >= 0.75:
        return 6.5, "Heavy edge/corner wear caps the grade."
    if wear_score >= 0.55:
        return 7.0, "Moderate-to-heavy edge/corner wear caps the grade."
    if wear_score >= 0.35:
        return 8.0, "Visible edge/corner wear caps the grade."
    if wear_score >= 0.2:
        return 9.0, "Light edge/corner wear applies a mild grade cap."
    return None, None


def _surface_penalty(surface_damage_score: float) -> tuple[float, str] | tuple[None, None]:
    if surface_damage_score >= 0.75:
        return 2.5, "Severe surface damage penalty."
    if surface_damage_score >= 0.55:
        return 2.0, "Major surface damage penalty."
    if surface_damage_score >= 0.35:
        return 1.5, "Moderate surface damage penalty."
    if surface_damage_score >= 0.2:
        return 1.0, "Minor-to-moderate surface damage penalty."
    if surface_damage_score >= 0.1:
        return 0.5, "Light surface issue penalty."
    return None, None


def _score_to_range(score: float) -> str:
    if score >= 9.5:
        return "9-10"
    if score >= 8.5:
        return "8-9"
    if score >= 7.5:
        return "7-8"
    if score >= 6.5:
        return "6-7"
    if score >= 5.5:
        return "5-6"
    return "1-5"


def estimate_grade(centering: dict, edge_corner: dict, surface: dict) -> dict:
    reasons: list[str] = []

    base_score, centering_reason = _centering_base_score(centering)
    reasons.append(centering_reason)

    wear_score = float(edge_corner.get("combined_wear_score", 0.0))
    wear_cap, wear_reason = _edge_wear_cap(wear_score)

    score_after_caps = base_score
    if wear_cap is not None:
        score_after_caps = min(score_after_caps, wear_cap)
    if wear_reason:
        reasons.append(wear_reason)

    surface_damage_score = float(surface.get("damage_score", 0.0))
    penalty, penalty_reason = _surface_penalty(surface_damage_score)

    final_score = score_after_caps
    if penalty is not None:
        final_score -= penalty
    if penalty_reason:
        reasons.append(penalty_reason)

    final_score = _clamp(final_score, 1.0, 10.0)

    estimated_range = _score_to_range(final_score)
    if final_score >= 8.8 and wear_score < 0.25 and surface_damage_score < 0.2:
        confidence = "high"
    elif final_score < 6.5 or wear_score > 0.6 or surface_damage_score > 0.5:
        confidence = "low"
    else:
        confidence = "medium"

    return {
        "estimated_score": round(final_score, 2),
        "estimated_range": estimated_range,
        "centering_base_score": round(base_score, 2),
        "edge_wear_cap": wear_cap,
        "surface_penalty": penalty or 0.0,
        "confidence": confidence,
        "reasons": reasons,
    }

