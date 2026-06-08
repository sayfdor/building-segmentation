"""
src/postprocess.py

Post-processing pipeline for predicted building masks.

Functions:
    apply_morphology — morphological cleanup (noise + holes);
    extract_building_polygons — contours -> simplified polygons;
    polygons_to_mask — render polygons -> binary mask;
    regularize_buildings — contours -> polygons -> mask;
    postprocess_pipeline — full pipeline: morphology + regularization.
"""
import cv2
import numpy as np


def apply_morphology(mask, kernel_size=3, close_iter=2, open_iter=1):
    mask_u8 = ((mask > 0).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return opened


def extract_building_polygons(binary_mask, min_area=40, simplify_frac=0.01):
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, simplify_frac * peri, True)
        if len(approx) < 3:
            continue  # вырожденный контур — не полигон
        polygons.append((approx.reshape(-1, 2), float(area)))
    return polygons


def polygons_to_mask(polygons, shape):
    out = np.zeros(shape[:2], dtype=np.uint8)
    for poly, _ in polygons:
        cv2.drawContours(out, [poly.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    return out


def regularize_buildings(binary_mask, min_area=40, simplify_frac=0.02):
    polys = extract_building_polygons(binary_mask, min_area=min_area, simplify_frac=simplify_frac)
    return polygons_to_mask(polys, binary_mask.shape)


def postprocess_pipeline(raw_binary, kernel_size=3, min_area=10, simplify_frac=0.01):
    morph = apply_morphology(raw_binary, kernel_size=kernel_size)
    polys = extract_building_polygons(morph, min_area=min_area, simplify_frac=simplify_frac)
    approx = polygons_to_mask(polys, morph.shape)
    return morph, approx
