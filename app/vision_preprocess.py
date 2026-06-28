from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    card_size: tuple[int, int]
    card_detected: bool
    contour_area_ratio: float
    resized_image: np.ndarray
    edge_image: np.ndarray
    card_image: np.ndarray


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return image


def resize_for_processing(image: np.ndarray, max_side: int = 1200) -> np.ndarray:
    height, width = image.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_side:
        return image.copy()

    scale = max_side / float(largest_dim)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    point_sums = points.sum(axis=1)
    rect[0] = points[np.argmin(point_sums)]  # top-left
    rect[2] = points[np.argmax(point_sums)]  # bottom-right

    point_diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(point_diff)]  # top-right
    rect[3] = points[np.argmax(point_diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    max_height = int(max(height_left, height_right))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def detect_card_contour(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contour_data = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]

    image_area = float(image.shape[0] * image.shape[1])
    best_ratio = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area_ratio = cv2.contourArea(approx) / image_area
        best_ratio = max(best_ratio, area_ratio)

        if len(approx) == 4 and area_ratio >= 0.2:
            return approx.reshape(4, 2).astype("float32"), edges, float(area_ratio)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / image_area
        best_ratio = max(best_ratio, area_ratio)

        if area_ratio >= 0.25:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect).astype("float32")
            return box, edges, float(area_ratio)

    return None, edges, float(best_ratio)


def normalize_orientation(card_image: np.ndarray) -> np.ndarray:
    height, width = card_image.shape[:2]
    if width > height:
        return cv2.rotate(card_image, cv2.ROTATE_90_CLOCKWISE)
    return card_image


def preprocess_card_image(image: np.ndarray, max_side: int = 1200) -> PreprocessResult:
    resized = resize_for_processing(image, max_side=max_side)
    contour, edge_image, contour_area_ratio = detect_card_contour(resized)

    if contour is not None:
        card = four_point_transform(resized, contour)
        card_detected = True
    else:
        card = resized.copy()
        card_detected = False

    card = normalize_orientation(card)
    card = resize_for_processing(card, max_side=900)

    original_size = (int(image.shape[1]), int(image.shape[0]))
    resized_size = (int(resized.shape[1]), int(resized.shape[0]))
    card_size = (int(card.shape[1]), int(card.shape[0]))

    return PreprocessResult(
        original_size=original_size,
        resized_size=resized_size,
        card_size=card_size,
        card_detected=card_detected,
        contour_area_ratio=round(contour_area_ratio, 4),
        resized_image=resized,
        edge_image=edge_image,
        card_image=card,
    )

