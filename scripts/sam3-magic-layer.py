#!/usr/bin/env python3
"""SAM 3 wrapper for BananaTape Magic Layer.

This script adapts Meta's official `facebookresearch/sam3` Python API to the
JSON contract expected by BananaTape's `/api/magic-layer` route.

Requirements are intentionally external to BananaTape's npm package:
  - Python 3.12+
  - a working SAM 3 installation (`pip install -e .` from facebookresearch/sam3)
  - accepted/downloadable SAM 3 checkpoints through the official mechanism
  - a PyTorch runtime supported by the SAM 3 package

Example:
  python3 scripts/sam3-magic-layer.py --input image.png --output segments.json \
    --prompts text,logo,person,product,object
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 3 and emit BananaTape Magic Layer segments JSON.")
    parser.add_argument("positional_input", nargs="?", help="Input image path (fallback positional form).")
    parser.add_argument("positional_output", nargs="?", help="Output JSON path (fallback positional form).")
    parser.add_argument("--input", dest="input_path", help="Input image path.")
    parser.add_argument("--output", dest="output_path", help="Output JSON path.")
    parser.add_argument(
        "--prompts",
        default="person,animal,cat,dog,bird,plant,flower,tree,food,product,bottle,cup,chair,table,car,building,text,logo",
        help="Comma-separated SAM 3 concept prompts to try. Specific nouns work much better than generic 'object'.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Minimum SAM 3 score to keep after model inference.")
    parser.add_argument("--max-segments", type=int, default=40, help="Maximum number of segments to emit.")
    parser.add_argument("--min-area-ratio", type=float, default=0.002, help="Drop masks smaller than this fraction of the image.")
    parser.add_argument("--max-area-ratio", type=float, default=0.75, help="Drop masks larger than this fraction of the image.")
    parser.add_argument("--nms-iou", type=float, default=0.72, help="Drop lower-scoring boxes with IoU above this value.")
    parser.add_argument("--layout-segments", type=int, default=12, help="Maximum coarse no-dependency layout/color components to add for slide-like images.")
    parser.add_argument("--layout-detail-segments", type=int, default=24, help="Maximum fine no-dependency layout/color components to add for slide-like images.")
    parser.add_argument("--layout-color-segments", type=int, default=24, help="Maximum color-separated layout components to add for slide-like images.")
    parser.add_argument("--layout-distance-threshold", type=float, default=42.0, help="RGB distance from border-estimated background for layout component extraction.")
    parser.add_argument("--ocr-provider", choices=("none", "tesseract", "paddle"), default="tesseract", help="Optional OCR provider for text-heavy Magic Layer boxes.")
    parser.add_argument("--ocr-segments", type=int, default=16, help="Maximum optional OCR text boxes to add for text-heavy images.")
    parser.add_argument("--ocr-max-layout-segments", type=int, default=6, help="When OCR is enabled, cap retained layout/group boxes so text boxes do not compete with many coarse duplicates. Use a negative value to disable this cap.")
    parser.add_argument("--ocr-min-confidence", type=float, default=70.0, help="Minimum OCR confidence for OCR text boxes.")
    parser.add_argument("--ocr-lang", default="kor+eng", help="Tesseract language pack for optional OCR text boxes.")
    parser.add_argument("--paddle-lang", default="korean", help="PaddleOCR language code for optional OCR text boxes.")
    parser.add_argument("--paddle-engine", choices=("paddle", "transformers"), default="paddle", help="PaddleOCR 3.x inference engine for optional OCR text boxes.")
    return parser.parse_args()


def to_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        tensor = value.detach()
        # NumPy cannot directly materialize torch.bfloat16 tensors. Convert
        # floating outputs to float32 before moving them to CPU/NumPy for JSON
        # serialization and mask post-processing.
        if getattr(tensor, "is_floating_point", lambda: False)():
            tensor = tensor.float()
        return tensor.cpu().numpy()
    return np.asarray(value)


def data_url_from_mask(mask) -> str:
    import numpy as np
    from PIL import Image

    array = to_numpy(mask)
    while array.ndim > 2:
        array = array.squeeze(axis=0) if array.shape[0] == 1 else array[0]
    alpha = (array > 0).astype("uint8") * 255
    rgba = np.zeros((*alpha.shape, 4), dtype="uint8")
    rgba[..., :3] = 255
    rgba[..., 3] = alpha
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def box_to_xywh(box) -> dict[str, float]:
    values = to_numpy(box).astype(float).flatten().tolist()
    if len(values) != 4:
        raise ValueError(f"Expected 4 box values, got {len(values)}")
    x1, y1, x2, y2 = [float(v) for v in values]
    return {"x": x1, "y": y1, "width": max(1.0, x2 - x1), "height": max(1.0, y2 - y1)}


def score_at(scores, index: int) -> float:
    if scores is None:
        return 1.0
    arr = to_numpy(scores)
    if arr.size == 0:
        return 1.0
    return float(arr.flat[index])


def binary_mask(mask) -> "object":
    arr = to_numpy(mask)
    while arr.ndim > 2:
        arr = arr.squeeze(axis=0) if arr.shape[0] == 1 else arr[0]
    return arr > 0


def mask_bbox(mask, fallback_box) -> dict[str, float]:
    import numpy as np

    mask_array = binary_mask(mask)
    ys, xs = np.nonzero(mask_array)
    if xs.size == 0 or ys.size == 0:
        return box_to_xywh(fallback_box)
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return {"x": x1, "y": y1, "width": max(1.0, x2 - x1), "height": max(1.0, y2 - y1)}


def area_ratio(box: dict[str, float], image_size: tuple[int, int]) -> float:
    target_w, target_h = image_size
    return (box["width"] * box["height"]) / max(1.0, float(target_w * target_h))


def prompt_min_area_ratio(prompt: str, default_min_area_ratio: float) -> float:
    # SAM 3's "line" prompt tends to return many decorative strokes and page
    # footer glyphs on slide/card layouts. Keep only line-sized edit targets
    # large enough to be useful, while allowing small icons/symbols through the
    # global threshold.
    if prompt.strip().lower() == "line":
        return max(default_min_area_ratio, 0.0015)
    return default_min_area_ratio


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / max(union, 1e-6)


def intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_area(box: dict[str, float]) -> float:
    return max(0.0, box["width"]) * max(0.0, box["height"])


def containment_ratio(child: dict[str, float], parent: dict[str, float]) -> float:
    return intersection_area(child, parent) / max(box_area(child), 1e-6)


def normalized_ocr_text(segment: dict) -> str:
    label = str(segment.get("label", ""))
    if label.startswith("text:"):
        label = label[5:]
    return "".join(label.lower().split())


def prune_contained_ocr_segments(segments: list[dict], containment_threshold: float = 0.82) -> list[dict]:
    # OCR engines can emit nested text candidates for the same visual text:
    # word, phrase, sentence, and paragraph boxes. For direct manipulation, keep
    # the smallest selectable text units and drop larger boxes that mostly just
    # contain already-detected OCR units.
    ordered = sorted(segments, key=lambda segment: box_area(segment["bbox"]))
    kept: list[dict] = []
    for segment in ordered:
        box = segment["bbox"]
        text = normalized_ocr_text(segment)
        contained_smaller = [
            existing
            for existing in kept
            if containment_ratio(existing["bbox"], box) >= containment_threshold
        ]
        if len(contained_smaller) >= 2:
            continue
        if contained_smaller:
            smaller_text = normalized_ocr_text(contained_smaller[0])
            if smaller_text and text and (smaller_text in text or text in smaller_text):
                continue
        kept.append(segment)
    return kept


def sort_ocr_segments_for_selection(segments: list[dict]) -> list[dict]:
    return sorted(
        segments,
        key=lambda item: (float(item.get("score", 0)), -box_area(item["bbox"])),
        reverse=True,
    )


def suppress_nested_sam_parts(segments: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for child in segments:
        if child.get("source") == "layout":
            kept.append(child)
            continue
        child_box = child["bbox"]
        child_area = box_area(child_box)
        suppress = False
        for parent in segments:
            if parent is child or parent.get("source") == "layout":
                continue
            parent_box = parent["bbox"]
            parent_area = box_area(parent_box)
            if parent_area < child_area * 1.35:
                continue
            if containment_ratio(child_box, parent_box) < 0.82:
                continue
            if float(parent.get("score", 0)) < float(child.get("score", 0)) * 0.35:
                continue
            suppress = True
            break
        if not suppress:
            kept.append(child)
    return kept


def suppress_text_like_layouts_when_ocr_exists(segments: list[dict], image_size: tuple[int, int]) -> list[dict]:
    ocr_segments = [segment for segment in segments if segment.get("source") in {"ocr", "paddle-ocr"}]
    if not ocr_segments:
        return segments

    _image_w, image_h = image_size
    kept: list[dict] = []
    for segment in segments:
        if segment.get("source") != "layout":
            kept.append(segment)
            continue
        box = segment["bbox"]
        # Text-like layout components are usually narrow bands around headings,
        # phrases, or chip rows. When OCR already exposes the smaller text units,
        # these layout bands mostly add duplicate click targets.
        if area_ratio(box, image_size) > 0.08 or (box["height"] / max(float(image_h), 1.0)) > 0.16:
            kept.append(segment)
            continue
        if any(containment_ratio(ocr["bbox"], box) >= 0.75 for ocr in ocr_segments):
            continue
        kept.append(segment)
    return kept


def cap_layout_segments_when_ocr_exists(segments: list[dict], max_layout_segments: int) -> list[dict]:
    if max_layout_segments < 0:
        return segments
    if not any(segment.get("source") in {"ocr", "paddle-ocr"} for segment in segments):
        return segments
    kept: list[dict] = []
    layout_count = 0
    for segment in segments:
        if segment.get("source") == "layout":
            if layout_count >= max_layout_segments:
                continue
            layout_count += 1
        kept.append(segment)
    return kept


def dedupe_segments(segments: list[dict], nms_iou: float, max_segments: int) -> list[dict]:
    candidates = suppress_nested_sam_parts(segments)
    ordered = sorted(
        candidates,
        key=lambda s: (float(s.get("score", 0)), min(box_area(s["bbox"]), 1_000_000.0)),
        reverse=True,
    )
    kept: list[dict] = []
    for segment in ordered:
        if any(iou(segment["bbox"], existing["bbox"]) >= nms_iou for existing in kept):
            continue
        kept.append(segment)
        if len(kept) >= max_segments:
            break
    return kept


def connected_components(mask: "object") -> list[tuple[int, int, int, int, int]]:
    import numpy as np

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for start_x, start_y in zip(xs.tolist(), ys.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True
        min_x = max_x = start_x
        min_y = max_y = start_y
        area = 0
        while stack:
            x, y = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= w or ny >= h or visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((nx, ny))
        components.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return components


def layout_component_segments(
    foreground,
    image_size: tuple[int, int],
    scale: float,
    kernel: int,
    max_segments: int,
    min_area_ratio: float,
    max_area_ratio: float,
    score: float,
    id_prefix: str,
    label: str,
    include_filled_box: bool = False,
) -> list[tuple[float, dict]]:
    import numpy as np
    from PIL import Image, ImageFilter

    if max_segments <= 0:
        return []

    target_w, target_h = image_size
    small_w = max(1, int(round(target_w * scale)))
    small_h = max(1, int(round(target_h * scale)))
    small = Image.fromarray((foreground.astype("uint8") * 255), mode="L").resize((small_w, small_h), resample=Image.Resampling.NEAREST)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        small = small.filter(ImageFilter.MaxFilter(kernel))
    small_mask = np.asarray(small) > 0

    components = []
    layout_max_area_ratio = min(max_area_ratio, 0.45)
    layout_min_area_ratio = min_area_ratio if id_prefix == "layout" else min(min_area_ratio, 0.00045)
    for sx1, sy1, sx2, sy2, small_area in connected_components(small_mask):
        x1 = max(0, int(sx1 / scale) - 3)
        y1 = max(0, int(sy1 / scale) - 3)
        x2 = min(target_w, int(np.ceil(sx2 / scale)) + 3)
        y2 = min(target_h, int(np.ceil(sy2 / scale)) + 3)
        if x2 <= x1 or y2 <= y1:
            continue
        component_mask = np.zeros((target_h, target_w), dtype=bool)
        if include_filled_box:
            component_mask[y1:y2, x1:x2] = True
        else:
            component_mask[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
        bbox = mask_bbox(component_mask, [x1, y1, x2, y2])
        ratio = area_ratio(bbox, image_size)
        if ratio < layout_min_area_ratio or ratio > layout_max_area_ratio:
            continue
        # Prefer practical movable elements over tiny glyph dust: very thin components
        # are usually individual strokes unless they are wide enough to be a text line.
        if id_prefix == "layout-detail" and (bbox["width"] < 8 or bbox["height"] < 8):
            continue
        priority = bbox["width"] * bbox["height"] + small_area / max(scale * scale, 1e-6)
        components.append((priority, bbox, component_mask))

    components.sort(key=lambda item: item[0], reverse=True)
    segments = []
    for emitted, (_priority, bbox, component_mask) in enumerate(components[:max_segments], start=1):
        segments.append((
            box_area(bbox),
            {
                "id": f"{id_prefix}-{emitted}",
                "label": label,
                "score": score,
                "source": "layout",
                "bbox": bbox,
                "maskDataUrl": data_url_from_mask(component_mask),
            },
        ))
    return segments


def color_component_segments(
    arr,
    foreground,
    image_size: tuple[int, int],
    scale: float,
    max_segments: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[tuple[float, dict]]:
    import numpy as np

    if max_segments <= 0:
        return []

    quantized = (arr // 32).astype("int16")
    flat_keys = quantized[..., 0] * 100 + quantized[..., 1] * 10 + quantized[..., 2]
    keys, counts = np.unique(flat_keys[foreground], return_counts=True)
    order = keys[np.argsort(counts)[::-1]]
    segments: list[tuple[float, dict]] = []
    for key in order.tolist():
        if len(segments) >= max_segments * 3:
            break
        color_mask = foreground & (flat_keys == key)
        if float(color_mask.mean()) < 0.0002:
            continue
        segments.extend(layout_component_segments(
            color_mask,
            image_size,
            scale,
            max(3, int(round(5 * scale))),
            max_segments,
            min_area_ratio,
            max_area_ratio,
            0.43,
            "layout-color",
            "layout color element",
            include_filled_box=False,
        ))
    segments.sort(key=lambda item: item[0], reverse=True)
    deduped: list[tuple[float, dict]] = []
    seen: set[str] = set()
    for _area, segment in segments:
        bbox = segment["bbox"]
        key = f"{round(bbox['x'])}:{round(bbox['y'])}:{round(bbox['width'])}:{round(bbox['height'])}"
        if key in seen:
            continue
        seen.add(key)
        segment["id"] = f"layout-color-{len(deduped) + 1}"
        deduped.append((box_area(bbox), segment))
        if len(deduped) >= max_segments:
            break
    return deduped


def iter_layout_segments(
    image,
    image_size: tuple[int, int],
    max_segments: int,
    detail_segments: int,
    color_segments: int,
    threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> Iterable[dict]:
    import numpy as np

    target_w, target_h = image_size
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    distance = np.sqrt(((arr - background) ** 2).sum(axis=2))
    foreground = distance > threshold
    foreground_ratio = float(foreground.mean())
    if foreground_ratio < 0.005 or foreground_ratio > 0.78:
        return

    scale = min(1.0, 420.0 / max(target_w, target_h))
    coarse_kernel = max(3, int(round(19 * scale)))
    detail_kernel = max(3, int(round(7 * scale)))

    coarse = layout_component_segments(
        foreground,
        image_size,
        scale,
        coarse_kernel,
        max_segments,
        min_area_ratio,
        max_area_ratio,
        0.45,
        "layout",
        "layout group",
        include_filled_box=True,
    )
    details = layout_component_segments(
        foreground,
        image_size,
        scale,
        detail_kernel,
        detail_segments,
        min_area_ratio,
        max_area_ratio,
        0.44,
        "layout-detail",
        "layout detail",
        include_filled_box=False,
    )
    colors = color_component_segments(
        arr,
        foreground,
        image_size,
        scale,
        color_segments,
        min_area_ratio,
        max_area_ratio,
    )

    emitted: set[str] = set()
    for _area, segment in coarse + details + colors:
        key = f"{round(segment['bbox']['x'])}:{round(segment['bbox']['y'])}:{round(segment['bbox']['width'])}:{round(segment['bbox']['height'])}"
        if key in emitted:
            continue
        emitted.add(key)
        yield segment


def iter_ocr_text_segments(input_path: Path, image_size: tuple[int, int], max_segments: int, min_confidence: float, lang: str, min_area_ratio: float) -> Iterable[dict]:
    if max_segments <= 0:
        return

    target_w, target_h = image_size
    try:
        with tempfile.TemporaryDirectory(prefix="bananatape-ocr-") as tmp:
            output_base = Path(tmp) / "ocr"
            subprocess.run(
                ["tesseract", str(input_path), str(output_base), "-l", lang, "--psm", "6", "tsv"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            tsv_path = output_base.with_suffix(".tsv")
            with tsv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
    except (FileNotFoundError, subprocess.SubprocessError, OSError, UnicodeDecodeError):
        return

    candidates: list[dict] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf", "-1"))
            x = int(float(row.get("left", "0")))
            y = int(float(row.get("top", "0")))
            width = int(float(row.get("width", "0")))
            height = int(float(row.get("height", "0")))
        except ValueError:
            continue
        if confidence < min_confidence or width < 12 or height < 12:
            continue
        pad = max(3, min(10, round(height * 0.12)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(target_w, x + width + pad)
        y2 = min(target_h, y + height + pad)
        bbox = {"x": float(x1), "y": float(y1), "width": float(max(1, x2 - x1)), "height": float(max(1, y2 - y1))}
        ratio = area_ratio(bbox, image_size)
        if ratio < min_area_ratio or ratio > 0.18:
            continue
        candidates.append({
            "id": f"ocr-word-{len(candidates) + 1}",
            "label": f"text: {text}",
            "score": max(0.0, min(1.0, confidence / 100.0)),
            "source": "ocr",
            "bbox": bbox,
        })

    candidates = sort_ocr_segments_for_selection(prune_contained_ocr_segments(candidates))
    for segment in candidates[:max_segments]:
        yield segment


def instantiate_paddle_ocr(lang: str, engine: str):
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(f"PaddleOCR dependencies are not available: {exc}") from exc

    init_attempts = [
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": engine,
        },
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "use_angle_cls": False,
            "show_log": False,
        },
        {
            "lang": lang,
            "use_angle_cls": False,
        },
    ]
    errors: list[str] = []
    for kwargs in init_attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError(f"PaddleOCR initialization failed: {' | '.join(errors)}")


def result_mapping(result) -> dict:
    if isinstance(result, dict):
        inner = result.get("res")
        return inner if isinstance(inner, dict) else result
    try:
        mapped = dict(result)
        inner = mapped.get("res")
        return inner if isinstance(inner, dict) else mapped
    except Exception:
        return {}


def polygon_bbox(poly, image_size: tuple[int, int]) -> dict[str, float] | None:
    import numpy as np

    target_w, target_h = image_size
    arr = np.asarray(poly, dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return None
    x1 = max(0, int(np.floor(arr[:, 0].min())))
    y1 = max(0, int(np.floor(arr[:, 1].min())))
    x2 = min(target_w, int(np.ceil(arr[:, 0].max())))
    y2 = min(target_h, int(np.ceil(arr[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": float(x1), "y": float(y1), "width": float(max(1, x2 - x1)), "height": float(max(1, y2 - y1))}


def iter_paddle_v3_segments(results, image_size: tuple[int, int], min_score: float) -> Iterable[dict]:
    for result in results or []:
        mapped = result_mapping(result)
        polys = mapped.get("rec_polys")
        if polys is None:
            polys = mapped.get("dt_polys")
        boxes = mapped.get("rec_boxes")
        texts = list(mapped.get("rec_texts") or [])
        scores = list(mapped.get("rec_scores") or mapped.get("dt_scores") or [])
        if polys is None and boxes is None:
            continue

        regions = list(polys if polys is not None else boxes)
        for index, region in enumerate(regions):
            score = float(scores[index]) if index < len(scores) else 1.0
            text = str(texts[index]).strip() if index < len(texts) else ""
            if score < min_score:
                continue
            bbox = polygon_bbox(region, image_size)
            if bbox is None:
                continue
            yield {
                "label": f"text: {text}" if text else "text",
                "score": max(0.0, min(1.0, score)),
                "source": "paddle-ocr",
                "bbox": bbox,
            }


def is_legacy_paddle_row(value) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    box, rec = value
    if not isinstance(box, (list, tuple)) or not box:
        return False
    if not isinstance(rec, (list, tuple)) or len(rec) < 2:
        return False
    return isinstance(rec[0], str)


def iter_legacy_paddle_rows(value) -> Iterable[tuple[object, str, float]]:
    if is_legacy_paddle_row(value):
        box, rec = value
        try:
            yield box, str(rec[0]), float(rec[1])
        except (TypeError, ValueError):
            return
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_legacy_paddle_rows(item)


def iter_paddle_legacy_segments(results, image_size: tuple[int, int], min_score: float) -> Iterable[dict]:
    for box, text, score in iter_legacy_paddle_rows(results):
        if score < min_score:
            continue
        bbox = polygon_bbox(box, image_size)
        if bbox is None:
            continue
        yield {
            "label": f"text: {text.strip()}" if text.strip() else "text",
            "score": max(0.0, min(1.0, score)),
            "source": "paddle-ocr",
            "bbox": bbox,
        }


def iter_paddle_text_segments(
    input_path: Path,
    image_size: tuple[int, int],
    max_segments: int,
    min_confidence: float,
    lang: str,
    engine: str,
    min_area_ratio: float,
) -> Iterable[dict]:
    if max_segments <= 0:
        return

    min_score = max(0.0, min(1.0, min_confidence / 100.0))
    ocr = instantiate_paddle_ocr(lang, engine)
    try:
        if hasattr(ocr, "predict"):
            raw_results = ocr.predict(str(input_path))
            candidates = list(iter_paddle_v3_segments(raw_results, image_size, min_score))
        else:
            raw_results = ocr.ocr(str(input_path), cls=False)
            candidates = list(iter_paddle_legacy_segments(raw_results, image_size, min_score))
    except Exception as exc:
        raise RuntimeError(f"PaddleOCR failed: {exc}") from exc

    filtered: list[dict] = []
    for index, segment in enumerate(candidates):
        bbox = segment["bbox"]
        ratio = area_ratio(bbox, image_size)
        if ratio < min_area_ratio or ratio > 0.18:
            continue
        filtered.append({
            **segment,
            "id": f"paddle-ocr-{index + 1}",
        })

    filtered = sort_ocr_segments_for_selection(prune_contained_ocr_segments(filtered))
    for segment in filtered[:max_segments]:
        yield segment


def iter_prompt_segments(processor, state, prompt: str, threshold: float, image_size: tuple[int, int], min_area_ratio: float, max_area_ratio: float) -> Iterable[dict]:
    output = processor.set_text_prompt(state=state, prompt=prompt)
    masks = output.get("masks", [])
    boxes = output.get("boxes", [])
    scores = output.get("scores")

    for index, (mask, box) in enumerate(zip(masks, boxes)):
        score = score_at(scores, index)
        if score < threshold:
            continue
        bbox = mask_bbox(mask, box)
        ratio = area_ratio(bbox, image_size)
        prompt_min_area = prompt_min_area_ratio(prompt, min_area_ratio)
        if ratio < prompt_min_area or ratio > max_area_ratio:
            continue
        yield {
            "id": f"{prompt.replace(' ', '-')}-{index + 1}",
            "label": prompt,
            "score": score,
            "source": "sam",
            "bbox": bbox,
            "maskDataUrl": data_url_from_mask(mask),
        }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path or args.positional_input or "")
    output_path = Path(args.output_path or args.positional_output or "")
    if not input_path.is_file() or not output_path:
        print("input and output paths are required", file=sys.stderr)
        return 2

    try:
        import torch
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:  # pragma: no cover - depends on external SAM 3 env
        print(f"SAM 3 Python dependencies are not available: {exc}", file=sys.stderr)
        return 3

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    image = Image.open(input_path).convert("RGB")

    # SAM 3's CUDA image path expects bf16 autocast around inference on recent
    # NVIDIA/PyTorch stacks. Without this, the ViT forward can mix BFloat16
    # activations with Float32 weights and fail with a dtype mismatch.
    inference_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    with inference_context:
        state = processor.set_image(image)

    candidates: list[dict] = []
    prompts = [prompt.strip() for prompt in args.prompts.split(",") if prompt.strip()]
    image_size = image.size
    with inference_context:
        for prompt in prompts:
            for segment in iter_prompt_segments(processor, state, prompt, args.score_threshold, image_size, args.min_area_ratio, args.max_area_ratio):
                candidates.append(segment)

    if args.layout_segments > 0:
        candidates.extend(iter_layout_segments(image, image_size, args.layout_segments, args.layout_detail_segments, args.layout_color_segments, args.layout_distance_threshold, args.min_area_ratio, args.max_area_ratio))
    if args.ocr_segments > 0 and args.ocr_provider == "tesseract":
        candidates.extend(iter_ocr_text_segments(input_path, image_size, args.ocr_segments, args.ocr_min_confidence, args.ocr_lang, args.min_area_ratio))
    if args.ocr_segments > 0 and args.ocr_provider == "paddle":
        candidates.extend(iter_paddle_text_segments(input_path, image_size, args.ocr_segments, args.ocr_min_confidence, args.paddle_lang, args.paddle_engine, args.min_area_ratio))

    candidates = suppress_text_like_layouts_when_ocr_exists(candidates, image_size)
    segments = dedupe_segments(candidates, args.nms_iou, args.max_segments)
    segments = cap_layout_segments_when_ocr_exists(segments, args.ocr_max_layout_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if segments else 4


if __name__ == "__main__":
    raise SystemExit(main())
