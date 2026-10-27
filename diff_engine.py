"""
diff_engine.py — Hybrid PDF & Image Comparator Engine
======================================================
Strategy (in priority order):
  1. Text words    — page.get_text("words") + font/size/color from "dict" spans
  2. Drawings      — page.get_drawings(), paired by bbox IoU
  3. Embedded imgs — page.get_image_info(), paired by IoU, hash-compared
  4. Residual pixels — remaining unmasked area, tight morphological diff

All handled regions are masked out of the pixel pass so anti-aliasing raster noise
on unchanged elements never fires false positives.

Highlight colour convention:
  Doc A: red   = removed,  amber = modified
  Doc B: green = added,    amber = modified
  Heatmap: fills exact rects + residual changed pixels
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import io
import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

logger = logging.getLogger(__name__)

try:
    import pymupdf as fitz
    _HAS_FITZ = True
except ImportError:
    try:
        import fitz
        _HAS_FITZ = True
    except ImportError:
        _HAS_FITZ = False


# ─────────────────────────────────────────────────────────────────────────────
# PDF & Overlay Comparison (User Script Functions)
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_image(pdf_path_or_doc, page_num: int, matrix_scale: float = 5.0) -> np.ndarray:
    """
    Render a single page of a PDF document at high resolution (Matrix 5x by default).
    Accepts a filepath (str), raw PDF bytes, or an already open fitz.Document.
    Returns an RGB uint8 numpy array.
    """
    if isinstance(pdf_path_or_doc, str):
        doc = fitz.open(pdf_path_or_doc)
        close_doc = True
    elif isinstance(pdf_path_or_doc, bytes):
        doc = fitz.open(stream=pdf_path_or_doc, filetype="pdf")
        close_doc = True
    else:
        doc = pdf_path_or_doc
        close_doc = False

    page = doc[page_num]
    pix = page.get_pixmap(
        matrix=fitz.Matrix(matrix_scale, matrix_scale),
        alpha=False
    )
    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )
    if close_doc:
        doc.close()
    return img


def compare_pdfs_overlay(pdf1, pdf2, bytes1: bytes | None = None, bytes2: bytes | None = None, target_page: int | None = None) -> list[str]:
    """
    Compare two PDF documents and save overlay image(s) to 'OverlayComparison/' where:
      - Common content is preserved
      - Removed / pixel differences are highlighted in Red
      - New items / additions are highlighted in Blue
      - Legend overlay banner is drawn
    """
    output_folder = "OverlayComparison"
    os.makedirs(output_folder, exist_ok=True)
    doc1 = fitz.open(stream=bytes1, filetype="pdf") if bytes1 else fitz.open(pdf1)
    doc2 = fitz.open(stream=bytes2, filetype="pdf") if bytes2 else fitz.open(pdf2)
    pages = min(len(doc1), len(doc2))
    pdf1_name = os.path.splitext(os.path.basename(pdf1))[0] if pdf1 else "Document_A"
    pdf2_name = os.path.splitext(os.path.basename(pdf2))[0] if pdf2 else "Document_B"
    saved_files = []

    page_range = [target_page - 1] if target_page is not None and 1 <= target_page <= pages else range(pages)

    for page_num in page_range:
        print(f"Processing Page {page_num + 1}")
        img1 = pdf_to_image(doc1, page_num)
        img2 = pdf_to_image(doc2, page_num)
        # Match sizes
        img2 = cv2.resize(
            img2,
            (img1.shape[1], img1.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        # Convert to grayscale
        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2GRAY
        )
        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2GRAY
        )
        # Extract non-white content
        _, mask1 = cv2.threshold(
            gray1,
            240,
            255,
            cv2.THRESH_BINARY_INV
        )
        _, mask2 = cv2.threshold(
            gray2,
            240,
            255,
            cv2.THRESH_BINARY_INV
        )
        # Common
        common_mask = cv2.bitwise_and(
            mask1,
            mask2
        )
        # Only in PDF1 (removed / pixel differences)
        removed_mask = cv2.bitwise_and(
            mask1,
            cv2.bitwise_not(mask2)
        )
        # Only in PDF2 (new items / additions)
        added_mask = cv2.bitwise_and(
            mask2,
            cv2.bitwise_not(mask1)
        )

        # Start with original drawing
        overlay = img1.copy()
        # Removed -> Red [255, 0, 0] in RGB
        overlay[removed_mask > 0] = [255, 0, 0]
        # Added -> Blue [0, 0, 255] in RGB
        overlay[added_mask > 0] = [0, 0, 255]

        # Legend
        cv2.rectangle(
            overlay,
            (10, 10),
            (650, 80),
            (255, 255, 255),
            -1
        )
        cv2.putText(
            overlay,
            "GRAY = COMMON",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (100, 100, 100),
            2
        )
        cv2.putText(
            overlay,
            "RED = REMOVED",
            (240, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )
        cv2.putText(
            overlay,
            "BLUE = ADDED",
            (470, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        output_file = os.path.join(
            output_folder,
            f"{pdf1_name}_VS_{pdf2_name}_Page_{page_num + 1}.png"
        )
        cv2.imwrite(
            output_file,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        print(
            f"Saved: {output_file}"
        )
        print(
            f"Removed={np.count_nonzero(removed_mask)}, "
            f"Added={np.count_nonzero(added_mask)}"
        )
        saved_files.append(output_file)

    doc1.close()
    doc2.close()
    print(
        f"\nResults saved in '{output_folder}'"
    )
    return saved_files


# ─────────────────────────────────────────────────────────────────────────────
# Rect & Geometric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou(r1: tuple[float, float, float, float], r2: tuple[float, float, float, float], pad: float = 2.0) -> float:
    """
    Intersection-over-union for two (x0, y0, x1, y1) rects.
    Padded by `pad` points so zero-width/height lines have area and can be paired reliably.
    """
    x0_1, y0_1, x1_1, y1_1 = r1[0] - pad, r1[1] - pad, r1[2] + pad, r1[3] + pad
    x0_2, y0_2, x1_2, y1_2 = r2[0] - pad, r2[1] - pad, r2[2] + pad, r2[3] + pad

    x0 = max(x0_1, x0_2)
    y0 = max(y0_1, y0_2)
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)

    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (x1_1 - x0_1)) * max(0.0, (y1_1 - y0_1))
    a2 = max(0.0, (x1_2 - x0_2)) * max(0.0, (y1_2 - y0_2))
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


def _rect_to_pixels(rect: tuple[float, float, float, float], scale: float,
                    max_w: int, max_h: int, pad: float = 0.0,
                    origin_x: float = 0.0, origin_y: float = 0.0) -> tuple[int, int, int, int]:
    """
    Convert PDF-points rect (x0, y0, x1, y1) to pixel bounds within (max_w, max_h),
    taking page origin offset into account.
    """
    rx0 = (rect[0] - origin_x - pad) * scale
    ry0 = (rect[1] - origin_y - pad) * scale
    rx1 = (rect[2] - origin_x + pad) * scale
    ry1 = (rect[3] - origin_y + pad) * scale

    x0 = max(0, min(max_w, int(np.floor(min(rx0, rx1)))))
    y0 = max(0, min(max_h, int(np.floor(min(ry0, ry1)))))
    x1 = max(x0, min(max_w, int(np.ceil(max(rx0, rx1)))))
    y1 = max(y0, min(max_h, int(np.ceil(max(ry0, ry1)))))
    return (x0, y0, x1, y1)


def _crop_similar(arr_a: np.ndarray, arr_b: np.ndarray,
                  px0: int, py0: int, px1: int, py1: int, tol: float = 14.0) -> bool:
    """Mean absolute RGB delta within a bounding box; True -> visually identical."""
    h, w = arr_a.shape[:2]
    cx0 = max(0, min(w, px0))
    cy0 = max(0, min(h, py0))
    cx1 = max(cx0, min(w, px1))
    cy1 = max(cy0, min(h, py1))
    if cx1 <= cx0 or cy1 <= cy0:
        return True
    ca = arr_a[cy0:cy1, cx0:cx1].astype(np.float32)
    cb = arr_b[cy0:cy1, cx0:cx1].astype(np.float32)
    return float(np.abs(ca - cb).mean()) <= tol


def _shift_tolerant_crop_similar(arr_a: np.ndarray, arr_b: np.ndarray,
                                  px0: int, py0: int, px1: int, py1: int,
                                  max_shift: int = 4, tol: float = 14.0) -> bool:
    """Check if crop region matches with a small shift (suppresses raster shifts)."""
    h, w = arr_a.shape[:2]
    for dy in range(-max_shift, max_shift + 1, 2):
        for dx in range(-max_shift, max_shift + 1, 2):
            bx0 = max(0, min(w, px0 + dx))
            by0 = max(0, min(h, py0 + dy))
            bx1 = max(bx0, min(w, px1 + dx))
            by1 = max(by0, min(h, py1 + dy))
            ax0 = max(0, min(w, px0))
            ay0 = max(0, min(h, py0))
            ax1 = max(ax0, min(w, px1))
            ay1 = max(ay0, min(h, py1))
            rw = min(ax1 - ax0, bx1 - bx0)
            rh = min(ay1 - ay0, by1 - by0)
            if rw <= 0 or rh <= 0:
                continue
            ca = arr_a[ay0:ay0 + rh, ax0:ax0 + rw].astype(np.float32)
            cb = arr_b[by0:by0 + rh, bx0:bx0 + rw].astype(np.float32)
            if float(np.abs(ca - cb).mean()) <= tol:
                return True
    return False


def image_to_b64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _convert_to_white_bg_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL Image to RGB mode, compositing transparency over a clean white canvas."""
    if img.mode == 'RGB':
        return img
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')


# ─────────────────────────────────────────────────────────────────────────────
# Data Class for Detected Changes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Change:
    x: int
    y: int
    width: int
    height: int
    change_type: str          # 'added' | 'removed' | 'modified'
    source: str               # 'text' | 'drawing' | 'image' | 'pixel'
    description: str = ''
    id: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Structured Text Comparison
# ─────────────────────────────────────────────────────────────────────────────

def _extract_spans(page) -> list[dict]:
    """Extract all text spans with font, size, and color from get_text('dict')."""
    spans = []
    try:
        text_dict = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in text_dict.get('blocks', []):
            if block.get('type') == 0:
                for line in block.get('lines', []):
                    for sp in line.get('spans', []):
                        spans.append({
                            'bbox': sp.get('bbox', (0, 0, 0, 0)),
                            'font': (sp.get('font') or '').split('+')[-1][:32],
                            'size': round(float(sp.get('size', 0)), 1),
                            'color': sp.get('color', 0),
                            'text': sp.get('text', ''),
                        })
    except Exception:
        logger.exception("Failed to extract spans from page")
    return spans


def _find_span_style(w_rect: tuple[float, float, float, float], spans: list[dict]) -> tuple[str, float, int]:
    """Find the font, size, and color of the span containing the word's center."""
    cx = (w_rect[0] + w_rect[2]) / 2.0
    cy = (w_rect[1] + w_rect[3]) / 2.0
    for sp in spans:
        sb = sp['bbox']
        if sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]:
            return (sp['font'], sp['size'], sp['color'])
    return ('', 0.0, 0)


def _compare_text_words(page_a, page_b, scale: float,
                        img_w: int, img_h: int) -> tuple[list[Change], np.ndarray, int]:
    """
    Word-level comparison:
      • Match identical words in close position & same style -> MASK rect (no noise)
      • Match same position with edited text or changed style -> MODIFIED word box
      • Match words moved far -> REMOVED at old box + ADDED at new box
      • Unmatched in A -> REMOVED word box
      • Unmatched in B -> ADDED word box
    Returns: (changes, handled_mask, matched_word_count)
    """
    changes: list[Change] = []
    handled_mask = np.zeros((img_h, img_w), dtype=bool)

    pw_a = max(float(page_a.rect.width or 0), 1.0)
    ph_a = max(float(page_a.rect.height or 0), 1.0)
    pw_b = max(float(page_b.rect.width or 0), 1.0)
    ph_b = max(float(page_b.rect.height or 0), 1.0)

    ox_a = float(page_a.rect.x0 or 0)
    oy_a = float(page_a.rect.y0 or 0)
    ox_b = float(page_b.rect.x0 or 0)
    oy_b = float(page_b.rect.y0 or 0)

    try:
        raw_words_a = page_a.get_text('words') or []
        raw_words_b = page_b.get_text('words') or []
    except Exception:
        logger.exception('Failed to extract PDF words; skipping text pass')
        return changes, handled_mask, 0

    spans_a = _extract_spans(page_a)
    spans_b = _extract_spans(page_b)

    # Build word records
    def build_records(raw_words, pw, ph, ox, oy, spans):
        records = []
        for w in raw_words:
            text = (w[4] or '').strip()
            if not text:
                continue
            r = (float(w[0]), float(w[1]), float(w[2]), float(w[3]))
            style = _find_span_style(r, spans)
            cx = (r[0] + r[2]) / 2.0
            cy = (r[1] + r[3]) / 2.0
            records.append({
                'text': text,
                'rect': r,
                'style': style,
                'cx_norm': (cx - ox) / pw,
                'cy_norm': (cy - oy) / ph,
                'matched': False,
            })
        return records

    words_a = build_records(raw_words_a, pw_a, ph_a, ox_a, oy_a, spans_a)
    words_b = build_records(raw_words_b, pw_b, ph_b, ox_b, oy_b, spans_b)

    matched_word_count = 0
    POS_THRESH = 0.04  # 4% of page dimension

    # Pass 1: Identical text words in close position
    for wa in words_a:
        if wa['matched']:
            continue
        best_b = None
        best_dist = float('inf')
        for wb in words_b:
            if wb['matched'] or wb['text'] != wa['text']:
                continue
            dist = ((wa['cx_norm'] - wb['cx_norm']) ** 2 + (wa['cy_norm'] - wb['cy_norm']) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_b = wb

        if best_b is not None and best_dist <= POS_THRESH:
            wa['matched'] = True
            best_b['matched'] = True
            matched_word_count += 1

            px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_a, origin_y=oy_a)
            px_b = _rect_to_pixels(best_b['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_b, origin_y=oy_b)

            # Mask both rects to eliminate anti-aliasing pixel noise
            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True

            # If formatting/font/color differs, record as modified
            if wa['style'] != best_b['style']:
                union_box = (
                    min(px_a[0], px_b[0]),
                    min(px_a[1], px_b[1]),
                    max(px_a[2], px_b[2]),
                    max(px_a[3], px_b[3]),
                )
                changes.append(Change(
                    x=union_box[0],
                    y=union_box[1],
                    width=max(4, union_box[2] - union_box[0]),
                    height=max(4, union_box[3] - union_box[1]),
                    change_type='modified',
                    source='text',
                    description=f'Word "{wa["text"]}" formatting or style changed.',
                ))

    # Pass 2: Words moved to a noticeably different location
    for wa in words_a:
        if wa['matched']:
            continue
        best_b = None
        best_dist = float('inf')
        for wb in words_b:
            if wb['matched'] or wb['text'] != wa['text']:
                continue
            dist = ((wa['cx_norm'] - wb['cx_norm']) ** 2 + (wa['cy_norm'] - wb['cy_norm']) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_b = wb

        if best_b is not None and best_dist > POS_THRESH:
            wa['matched'] = True
            best_b['matched'] = True
            matched_word_count += 1

            px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_a, origin_y=oy_a)
            px_b = _rect_to_pixels(best_b['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_b, origin_y=oy_b)

            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True

            changes.append(Change(
                x=px_a[0],
                y=px_a[1],
                width=max(4, px_a[2] - px_a[0]),
                height=max(4, px_a[3] - px_a[1]),
                change_type='removed',
                source='text',
                description=f'Word "{wa["text"]}" moved from this location.',
            ))
            changes.append(Change(
                x=px_b[0],
                y=px_b[1],
                width=max(4, px_b[2] - px_b[0]),
                height=max(4, px_b[3] - px_b[1]),
                change_type='added',
                source='text',
                description=f'Word "{best_b["text"]}" moved to this location.',
            ))

    # Pass 3: Words at roughly the same position with changed text (word replacement)
    for wa in words_a:
        if wa['matched']:
            continue
        best_b = None
        best_dist = float('inf')
        for wb in words_b:
            if wb['matched']:
                continue
            dist = ((wa['cx_norm'] - wb['cx_norm']) ** 2 + (wa['cy_norm'] - wb['cy_norm']) ** 2) ** 0.5
            if dist < POS_THRESH and dist < best_dist:
                best_dist = dist
                best_b = wb

        if best_b is not None:
            wa['matched'] = True
            best_b['matched'] = True

            px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_a, origin_y=oy_a)
            px_b = _rect_to_pixels(best_b['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_b, origin_y=oy_b)

            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True

            union_box = (
                min(px_a[0], px_b[0]),
                min(px_a[1], px_b[1]),
                max(px_a[2], px_b[2]),
                max(px_a[3], px_b[3]),
            )
            changes.append(Change(
                x=union_box[0],
                y=union_box[1],
                width=max(4, union_box[2] - union_box[0]),
                height=max(4, union_box[3] - union_box[1]),
                change_type='modified',
                source='text',
                description=f'Word "{wa["text"]}" changed to "{best_b["text"]}".',
            ))

    # Pass 4: Unmatched words in A -> removed
    for wa in words_a:
        if wa['matched']:
            continue
        px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_a, origin_y=oy_a)
        handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
        changes.append(Change(
            x=px_a[0],
            y=px_a[1],
            width=max(4, px_a[2] - px_a[0]),
            height=max(4, px_a[3] - px_a[1]),
            change_type='removed',
            source='text',
            description=f'Word "{wa["text"]}" removed.',
        ))

    # Pass 5: Unmatched words in B -> added
    for wb in words_b:
        if wb['matched']:
            continue
        px_b = _rect_to_pixels(wb['rect'], scale, img_w, img_h, pad=1.0, origin_x=ox_b, origin_y=oy_b)
        handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True
        changes.append(Change(
            x=px_b[0],
            y=px_b[1],
            width=max(4, px_b[2] - px_b[0]),
            height=max(4, px_b[3] - px_b[1]),
            change_type='added',
            source='text',
            description=f'Word "{wb["text"]}" added.',
        ))

    return changes, handled_mask, matched_word_count


# ─────────────────────────────────────────────────────────────────────────────
# Structured Drawing & Diagram Comparison
# ─────────────────────────────────────────────────────────────────────────────

def _drawing_sig(d: dict) -> str:
    """Compact signature of a drawing path for similarity comparison."""
    items = d.get('items') or []
    color = d.get('color')
    fill = d.get('fill')
    width = d.get('width')

    def item_str(it):
        try:
            kind = it[0]
            if kind == 'l':
                return f"l({round(it[1][0],1)},{round(it[1][1],1)},{round(it[2][0],1)},{round(it[2][1],1)})"
            if kind == 're':
                r = it[1]
                return f"re({round(r[0],1)},{round(r[1],1)},{round(r[2],1)},{round(r[3],1)})"
            if kind == 'c':
                pts = it[1:]
                return 'c(' + ','.join(f"{round(p[0],1)},{round(p[1],1)}" for p in pts if hasattr(p, '__len__')) + ')'
            return str(kind)
        except (IndexError, TypeError, ValueError):
            return '?'

    sig_items = ';'.join(item_str(it) for it in items[:15])
    return f"{color}|{fill}|{width}|{sig_items}"


def _compare_drawings(page_a, page_b, arr_a: np.ndarray, arr_b: np.ndarray,
                      scale: float, sensitivity: int,
                      img_w: int, img_h: int) -> tuple[list[Change], np.ndarray]:
    """
    Compare PDF vector drawings / diagrams:
      • Pair drawings by bbox IoU
      • High IoU + matching path signature / crop similarity -> IDENTICAL (mask, no false positives)
      • High IoU + different path or crop -> MODIFIED drawing box
      • Unpaired drawings -> ADDED / REMOVED drawing box
    """
    changes: list[Change] = []
    handled_mask = np.zeros((img_h, img_w), dtype=bool)

    ox_a = float(page_a.rect.x0 or 0)
    oy_a = float(page_a.rect.y0 or 0)
    ox_b = float(page_b.rect.x0 or 0)
    oy_b = float(page_b.rect.y0 or 0)

    try:
        draw_a = page_a.get_drawings() or []
        draw_b = page_b.get_drawings() or []
    except Exception:
        logger.exception('Failed to extract PDF drawings; skipping drawing pass')
        return changes, handled_mask

    if not draw_a and not draw_b:
        return changes, handled_mask

    def get_rect(d):
        r = d.get('rect')
        if r is None:
            return None
        return (float(r[0]), float(r[1]), float(r[2]), float(r[3]))

    matched_b: set[int] = set()

    for da in draw_a:
        ra = get_rect(da)
        if ra is None:
            continue
        best_iou = 0.0
        best_bi = -1
        for bi, db in enumerate(draw_b):
            if bi in matched_b:
                continue
            rb = get_rect(db)
            if rb is None:
                continue
            iou = _iou(ra, rb, pad=2.0)
            if iou > best_iou:
                best_iou = iou
                best_bi = bi

        stroke_pad_a = max(1.5, float(da.get('width') or 1.0) / 2.0)
        px_a = _rect_to_pixels(ra, scale, img_w, img_h, pad=stroke_pad_a, origin_x=ox_a, origin_y=oy_a)

        if best_iou >= 0.40 and best_bi >= 0:
            matched_b.add(best_bi)
            db = draw_b[best_bi]
            rb = get_rect(db)
            stroke_pad_b = max(1.5, float(db.get('width') or 1.0) / 2.0)
            px_b = _rect_to_pixels(rb, scale, img_w, img_h, pad=stroke_pad_b, origin_x=ox_b, origin_y=oy_b)

            sig_a = _drawing_sig(da)
            sig_b = _drawing_sig(db)
            is_same = (sig_a == sig_b)

            if not is_same:
                is_same = _shift_tolerant_crop_similar(
                    arr_a, arr_b,
                    min(px_a[0], px_b[0]), min(px_a[1], px_b[1]),
                    max(px_a[2], px_b[2]), max(px_a[3], px_b[3]),
                    max_shift=4, tol=float(sensitivity) * 1.5,
                )

            # Generous padding to suppress line edge AA noise
            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True

            if not is_same:
                union_box = (
                    min(px_a[0], px_b[0]),
                    min(px_a[1], px_b[1]),
                    max(px_a[2], px_b[2]),
                    max(px_a[3], px_b[3]),
                )
                changes.append(Change(
                    x=union_box[0],
                    y=union_box[1],
                    width=max(6, union_box[2] - union_box[0]),
                    height=max(6, union_box[3] - union_box[1]),
                    change_type='modified',
                    source='drawing',
                    description='Drawing or diagram element modified.',
                ))
        else:
            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            changes.append(Change(
                x=px_a[0],
                y=px_a[1],
                width=max(6, px_a[2] - px_a[0]),
                height=max(6, px_a[3] - px_a[1]),
                change_type='removed',
                source='drawing',
                description='Drawing or diagram element removed.',
            ))

    # Remaining unmatched drawings in B -> added
    for bi, db in enumerate(draw_b):
        if bi in matched_b:
            continue
        rb = get_rect(db)
        if rb is None:
            continue
        stroke_pad_b = max(1.5, float(db.get('width') or 1.0) / 2.0)
        px_b = _rect_to_pixels(rb, scale, img_w, img_h, pad=stroke_pad_b, origin_x=ox_b, origin_y=oy_b)
        handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True
        changes.append(Change(
            x=px_b[0],
            y=px_b[1],
            width=max(6, px_b[2] - px_b[0]),
            height=max(6, px_b[3] - px_b[1]),
            change_type='added',
            source='drawing',
            description='Drawing or diagram element added.',
        ))

    return changes, handled_mask


# ─────────────────────────────────────────────────────────────────────────────
# Embedded Image Comparison
# ─────────────────────────────────────────────────────────────────────────────

def _compare_images(page_a, page_b, arr_a: np.ndarray, arr_b: np.ndarray,
                    scale: float, sensitivity: int,
                    img_w: int, img_h: int) -> tuple[list[Change], np.ndarray]:
    """Pair embedded images by bbox IoU and compare hashes or crop similarity."""
    changes: list[Change] = []
    handled_mask = np.zeros((img_h, img_w), dtype=bool)

    ox_a = float(page_a.rect.x0 or 0)
    oy_a = float(page_a.rect.y0 or 0)
    ox_b = float(page_b.rect.x0 or 0)
    oy_b = float(page_b.rect.y0 or 0)

    try:
        imgs_a = page_a.get_image_info(xrefs=True) or []
        imgs_b = page_b.get_image_info(xrefs=True) or []
    except Exception:
        logger.exception('Failed to extract embedded images; skipping image pass')
        return changes, handled_mask

    if not imgs_a and not imgs_b:
        return changes, handled_mask

    matched_b: set[int] = set()

    for ia in imgs_a:
        bbox_a = ia.get('bbox')
        if not bbox_a:
            continue
        ra = (float(bbox_a[0]), float(bbox_a[1]), float(bbox_a[2]), float(bbox_a[3]))
        best_iou = 0.0
        best_bi = -1
        for bi, ib in enumerate(imgs_b):
            if bi in matched_b:
                continue
            bbox_b = ib.get('bbox')
            if not bbox_b:
                continue
            rb = (float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3]))
            iou = _iou(ra, rb, pad=2.0)
            if iou > best_iou:
                best_iou = iou
                best_bi = bi

        px_a = _rect_to_pixels(ra, scale, img_w, img_h, origin_x=ox_a, origin_y=oy_a)

        if best_iou >= 0.45 and best_bi >= 0:
            matched_b.add(best_bi)
            ib = imgs_b[best_bi]
            bbox_b = ib.get('bbox')
            rb = (float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3]))
            px_b = _rect_to_pixels(rb, scale, img_w, img_h, origin_x=ox_b, origin_y=oy_b)

            same = False
            if ia.get('xref') and ib.get('xref') and ia.get('xref') == ib.get('xref'):
                same = True
            elif ia.get('digest') and ib.get('digest') and ia.get('digest') == ib.get('digest'):
                same = True
            else:
                same = _shift_tolerant_crop_similar(
                    arr_a, arr_b,
                    min(px_a[0], px_b[0]), min(px_a[1], px_b[1]),
                    max(px_a[2], px_b[2]), max(px_a[3], px_b[3]),
                    max_shift=4, tol=float(sensitivity) * 1.5,
                )

            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True

            if not same:
                union_box = (
                    min(px_a[0], px_b[0]),
                    min(px_a[1], px_b[1]),
                    max(px_a[2], px_b[2]),
                    max(px_a[3], px_b[3]),
                )
                changes.append(Change(
                    x=union_box[0],
                    y=union_box[1],
                    width=max(6, union_box[2] - union_box[0]),
                    height=max(6, union_box[3] - union_box[1]),
                    change_type='modified',
                    source='image',
                    description='Embedded image modified.',
                ))
        else:
            handled_mask[px_a[1]:px_a[3], px_a[0]:px_a[2]] = True
            changes.append(Change(
                x=px_a[0],
                y=px_a[1],
                width=max(6, px_a[2] - px_a[0]),
                height=max(6, px_a[3] - px_a[1]),
                change_type='removed',
                source='image',
                description='Embedded image removed.',
            ))

    for bi, ib in enumerate(imgs_b):
        if bi in matched_b:
            continue
        bbox_b = ib.get('bbox')
        if not bbox_b:
            continue
        rb = (float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3]))
        px_b = _rect_to_pixels(rb, scale, img_w, img_h, origin_x=ox_b, origin_y=oy_b)
        handled_mask[px_b[1]:px_b[3], px_b[0]:px_b[2]] = True
        changes.append(Change(
            x=px_b[0],
            y=px_b[1],
            width=max(6, px_b[2] - px_b[0]),
            height=max(6, px_b[3] - px_b[1]),
            change_type='added',
            source='image',
            description='Embedded image added.',
        ))

    return changes, handled_mask


# ─────────────────────────────────────────────────────────────────────────────
# Residual Pixel Pass (Tight morphological diff for leftover elements)
# ─────────────────────────────────────────────────────────────────────────────

def _residual_pixel_diff(arr_a: np.ndarray, arr_b: np.ndarray,
                         handled_mask: np.ndarray, sensitivity: int) -> tuple[list[Change], np.ndarray]:
    """
    Diff remaining unmasked pixels:
      • binary_opening to drop 1px anti-aliasing specks
      • Dilate ONCE (1x), not 3x
      • Fast component extraction via ndimage.find_objects
      • Shift check to eliminate shifted raster artifacts
    """
    changes: list[Change] = []
    h, w = arr_a.shape[:2]

    diff_raw = (np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16)).mean(axis=2) > sensitivity)
    diff_unhandled = diff_raw & (~handled_mask)

    struct_3x3 = ndimage.generate_binary_structure(2, 2)
    opened = ndimage.binary_opening(diff_unhandled, structure=struct_3x3)
    dilated = ndimage.binary_dilation(opened, structure=struct_3x3, iterations=1)

    labeled, num_features = ndimage.label(dilated)
    if num_features == 0:
        return changes, opened

    slices = ndimage.find_objects(labeled)
    # Cap to avoid freezing on massive noise bursts
    max_components = 250
    if len(slices) > max_components:
        valid_slices = [s for s in slices if s is not None]
        valid_slices.sort(key=lambda sl: (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start), reverse=True)
        slices_to_process = valid_slices[:max_components]
    else:
        slices_to_process = [s for s in slices if s is not None]

    for sl in slices_to_process:
        sl_y, sl_x = sl
        min_y, max_y = sl_y.start, sl_y.stop - 1
        min_x, max_x = sl_x.start, sl_x.stop - 1
        cw = max_x - min_x + 1
        ch = max_y - min_y + 1

        if cw <= 1 and ch <= 1:
            continue

        active_in_slice = opened[sl_y, sl_x]
        if not np.any(active_in_slice) or int(active_in_slice.sum()) < 4:
            continue

        if _shift_tolerant_crop_similar(arr_a, arr_b, min_x, min_y, max_x + 1, max_y + 1,
                                         max_shift=4, tol=float(sensitivity) * 1.5):
            continue

        val_a = arr_a[min_y:max_y + 1, min_x:max_x + 1].mean()
        val_b = arr_b[min_y:max_y + 1, min_x:max_x + 1].mean()

        if val_a < 235 and val_b >= 215:
            change_type = 'removed'
            desc = f'Visual element removed at (X: {min_x}, Y: {min_y}, Size: {cw}×{ch}px).'
        elif val_a >= 215 and val_b < 235:
            change_type = 'added'
            desc = f'Visual element added at (X: {min_x}, Y: {min_y}, Size: {cw}×{ch}px).'
        else:
            change_type = 'modified'
            desc = f'Visual content modified at (X: {min_x}, Y: {min_y}, Size: {cw}×{ch}px).'

        changes.append(Change(
            x=min_x,
            y=min_y,
            width=cw,
            height=ch,
            change_type=change_type,
            source='pixel',
            description=desc,
        ))

    return changes, opened


# ─────────────────────────────────────────────────────────────────────────────
# High-Quality Rendering & PDF/Image Alignment
# ─────────────────────────────────────────────────────────────────────────────

def _render_pdf_page(page, scale: float) -> np.ndarray:
    """Render a PDF page to a uint8 RGB numpy array at given scale."""
    mat = fitz.Matrix(scale, scale)
    try:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    except TypeError:
        pix = page.get_pixmap(matrix=mat, alpha=False)
    if pix.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    samples = np.frombuffer(pix.samples, dtype=np.uint8)
    stride = pix.stride
    row_bytes = pix.width * pix.n
    if stride == row_bytes:
        arr = samples.reshape(pix.height, pix.width, pix.n)
    else:
        arr = samples.reshape(pix.height, stride)[:, :row_bytes].reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(arr)


def _pad_bottom_right(arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad both images bottom/right with white so they share identical dimensions."""
    ha, wa = arr_a.shape[:2]
    hb, wb = arr_b.shape[:2]
    H = max(ha, hb)
    W = max(wa, wb)
    def pad(arr, h, w):
        if h == H and w == W:
            return arr
        out = np.full((H, W, 3), 255, dtype=np.uint8)
        out[:h, :w] = arr
        return out
    return pad(arr_a, ha, wa), pad(arr_b, hb, wb)


def _open_pdf(file_bytes: bytes):
    """Open and return a fitz.Document."""
    pdf_start = file_bytes.find(b'%PDF-')
    clean = file_bytes[pdf_start:] if pdf_start >= 0 else file_bytes
    try:
        doc = fitz.open(stream=clean, filetype='pdf')
    except Exception:
        try:
            doc = fitz.open(stream=file_bytes, filetype='pdf')
        except Exception as exc:
            raise ValueError('The PDF could not be opened. It may be damaged or unreadable.') from exc
    if doc.needs_pass:
        authed = False
        try:
            authed = bool(doc.authenticate(''))
        except Exception:
            authed = False
        if not authed or doc.needs_pass:
            try:
                doc.close()
            except Exception:
                pass
            raise ValueError('This PDF is password-protected and cannot be compared.')
    if doc.page_count <= 0:
        try:
            doc.close()
        except Exception:
            pass
        raise ValueError('The PDF document contains 0 readable pages.')
    return doc


def _load_image(payload: bytes, filename: str, page: int = 1, max_dim: int = 1400) -> tuple[Image.Image, int]:
    """
    Safely load either a PDF page or a raster image into a PIL RGB Image.
    Returns (Image.Image in RGB, total_pages_count).
    """
    ext = (filename or '').lower()
    is_pdf = ext.endswith('.pdf') or (b'%PDF-' in payload[:4096])

    if is_pdf and _HAS_FITZ:
        doc = None
        try:
            doc = _open_pdf(payload)
            total_pages = max(1, doc.page_count)
            idx = max(0, min(page - 1, total_pages - 1))
            page_obj = doc.load_page(idx)
            pw = max(float(page_obj.rect.width or 0), 1.0)
            ph = max(float(page_obj.rect.height or 0), 1.0)
            scale = min(2.0, float(max_dim) / max(pw, ph))
            scale = max(0.05, scale)
            arr = _render_pdf_page(page_obj, scale)
            img = Image.fromarray(arr, 'RGB')
            return img, total_pages
        except Exception as exc:
            try:
                img = Image.open(io.BytesIO(payload))
                img = ImageOps.exif_transpose(img)
                return _convert_to_white_bg_rgb(img), 1
            except Exception:
                if isinstance(exc, ValueError):
                    raise
                raise ValueError(f'Failed to open document: {exc}') from exc
        finally:
            if doc is not None and not doc.is_closed:
                try:
                    doc.close()
                except Exception:
                    pass
    else:
        try:
            img = Image.open(io.BytesIO(payload))
            img = ImageOps.exif_transpose(img)
            return _convert_to_white_bg_rgb(img), 1
        except Exception as img_err:
            if _HAS_FITZ:
                try:
                    doc = _open_pdf(payload)
                    total_pages = max(1, doc.page_count)
                    idx = max(0, min(page - 1, total_pages - 1))
                    page_obj = doc.load_page(idx)
                    pw = max(float(page_obj.rect.width or 0), 1.0)
                    ph = max(float(page_obj.rect.height or 0), 1.0)
                    scale = min(2.0, float(max_dim) / max(pw, ph))
                    scale = max(0.05, scale)
                    arr = _render_pdf_page(page_obj, scale)
                    doc.close()
                    return Image.fromarray(arr, 'RGB'), total_pages
                except Exception:
                    pass
            raise ValueError('The uploaded file could not be decoded as an image or PDF.') from img_err


# ─────────────────────────────────────────────────────────────────────────────
# Visual Output Generation (Highlights, Heatmap, 3-Color, Snippets)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_visuals(img_a_clean: Image.Image, img_b_clean: Image.Image,
                      arr_a: np.ndarray, arr_b: np.ndarray,
                      changes: list[Change], changed_pixels_mask: np.ndarray) -> dict[str, Any]:
    """
    Generate all visual overlays:
      • Doc A highlighted: red boxes  = removed content  /  red+amber = modified
      • Doc B highlighted: blue boxes = added content    /  blue+amber = modified
      • Heatmap: grayed-down context + highlighted diff rects & residual pixels
      • 3-Color:  red=removed, blue=added, amber=modified
      • Thumbnail snippets for inspector cards

    IMPORTANT: Side-by-side highlights are driven exclusively by the structured
    changes list (bounding boxes from text/drawing/pixel passes).  We do NOT
    paint raw grayscale-mask differences onto the SBS overlays, because that
    causes the entire page to be colored for documents that share similar layouts
    but different content (e.g., two resumes or two invoices).
    """
    w, h = img_a_clean.size

    # Sort boxes in reading order: top-to-bottom, left-to-right
    changes.sort(key=lambda c: (c.y // 30, c.x))
    for i, c in enumerate(changes):
        c.id = i + 1

    # ── 1. Side-by-Side Highlighted Overlays ──────────────────────────────────
    # Use RGBA overlay on top of the clean original images so we can draw
    # semi-transparent colored rectangles precisely around detected change regions.
    #
    # Color convention (spec):
    #   removed  → red    on Doc A  (content that was in A, absent from B)
    #   added    → blue   on Doc B  (content that is new in B, absent from A)
    #   modified → red    on Doc A  AND  blue on Doc B  (changed at same location)
    ov_a = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov_b = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw_oa = ImageDraw.Draw(ov_a)
    draw_ob = ImageDraw.Draw(ov_b)

    RED_FILL    = (239, 68, 68, 55)
    RED_OUT     = (220, 38, 38, 230)
    BLUE_FILL   = (37, 99, 235, 55)
    BLUE_OUT    = (29, 78, 216, 230)
    AMBER_FILL  = (245, 158, 11, 55)
    AMBER_OUT   = (217, 119, 6, 230)

    for c in changes:
        x0 = max(0, min(w - 1, c.x - 2))
        y0 = max(0, min(h - 1, c.y - 2))
        x1 = max(x0 + 1, min(w, c.x + c.width + 2))
        y1 = max(y0 + 1, min(h, c.y + c.height + 2))
        radius = min(4, max(1, (x1 - x0) // 4, (y1 - y0) // 4))

        if c.change_type == 'removed':
            # Content existed in A but not B → red box on A only
            draw_oa.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                      fill=RED_FILL, outline=RED_OUT, width=2)
        elif c.change_type == 'added':
            # Content is new in B, absent from A → blue box on B only
            draw_ob.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                      fill=BLUE_FILL, outline=BLUE_OUT, width=2)
        else:
            # Modified at the same location: red on A, blue on B
            draw_oa.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                      fill=RED_FILL, outline=RED_OUT, width=2)
            draw_ob.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                      fill=BLUE_FILL, outline=BLUE_OUT, width=2)

    img_a_highlighted = Image.alpha_composite(img_a_clean.convert('RGBA'), ov_a).convert('RGB')
    img_b_highlighted = Image.alpha_composite(img_b_clean.convert('RGBA'), ov_b).convert('RGB')

    # For the heatmap & 3-color views, compute pixel-level removed/added masks
    # (these views CAN show per-pixel coloring; the SBS view must not).
    gray_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2GRAY)
    _, mask_content_a = cv2.threshold(gray_a, 240, 255, cv2.THRESH_BINARY_INV)
    _, mask_content_b = cv2.threshold(gray_b, 240, 255, cv2.THRESH_BINARY_INV)

    # Restrict pixel masks to only the regions flagged by changed_pixels_mask
    # so we don't paint the entire document when content differs everywhere.
    if changed_pixels_mask is not None and np.any(changed_pixels_mask):
        removed_pixel_mask = changed_pixels_mask & (mask_content_a > 0) & (mask_content_b == 0)
        added_pixel_mask   = changed_pixels_mask & (mask_content_b > 0) & (mask_content_a == 0)
        modified_pixel_mask = changed_pixels_mask & (mask_content_a > 0) & (mask_content_b > 0)
    else:
        removed_pixel_mask  = np.zeros((h, w), dtype=bool)
        added_pixel_mask    = np.zeros((h, w), dtype=bool)
        modified_pixel_mask = np.zeros((h, w), dtype=bool)

    # ── 2. High-Contrast Heatmap Diff ─────────────────────────────────────────
    # Gray-down the original A, then paint changed regions bright red.
    base_a = arr_a.copy()
    diff_canvas = np.clip(base_a.astype(np.float32) * 0.78 + 52, 0, 255).astype(np.uint8)

    # Fill change bounding boxes with a light-red tint
    for c in changes:
        cx0 = max(0, min(w, c.x))
        cy0 = max(0, min(h, c.y))
        cx1 = max(cx0, min(w, c.x + c.width))
        cy1 = max(cy0, min(h, c.y + c.height))
        if cx1 > cx0 and cy1 > cy0:
            diff_canvas[cy0:cy1, cx0:cx1] = [254, 226, 226]

    # Paint residual changed pixels bright red
    residual_mask = removed_pixel_mask | added_pixel_mask | modified_pixel_mask
    if np.any(residual_mask):
        diff_canvas[residual_mask] = [220, 38, 38]

    diff_img = Image.fromarray(diff_canvas, 'RGB')
    draw_diff = ImageDraw.Draw(diff_img)
    for c in changes:
        x0 = max(0, min(w - 1, c.x - 2))
        y0 = max(0, min(h - 1, c.y - 2))
        x1 = max(x0 + 1, min(w, c.x + c.width + 2))
        y1 = max(y0 + 1, min(h, c.y + c.height + 2))
        radius = min(3, max(1, (x1 - x0) // 2, (y1 - y0) // 2))
        draw_diff.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=(220, 38, 38), width=2)

    # ── 3. 3-Color Visual Diff (Removed=Red / Added=Blue / Modified=Amber) ────
    diff_3c = np.clip(base_a.astype(np.float32) * 0.78 + 52, 0, 255).astype(np.uint8)

    for c in changes:
        cx0 = max(0, min(w, c.x))
        cy0 = max(0, min(h, c.y))
        cx1 = max(cx0, min(w, c.x + c.width))
        cy1 = max(cy0, min(h, c.y + c.height))
        if cx1 > cx0 and cy1 > cy0:
            if c.change_type == 'removed':
                diff_3c[cy0:cy1, cx0:cx1] = [239, 68, 68]
            elif c.change_type == 'added':
                diff_3c[cy0:cy1, cx0:cx1] = [37, 99, 235]
            else:
                diff_3c[cy0:cy1, cx0:cx1] = [245, 158, 11]

    if np.any(removed_pixel_mask):
        diff_3c[removed_pixel_mask] = [239, 68, 68]
    if np.any(added_pixel_mask):
        diff_3c[added_pixel_mask] = [37, 99, 235]
    if np.any(modified_pixel_mask):
        diff_3c[modified_pixel_mask] = [245, 158, 11]

    diff_3c_img = Image.fromarray(diff_3c, 'RGB')
    draw_3c = ImageDraw.Draw(diff_3c_img)
    for c in changes:
        x0 = max(0, min(w - 1, c.x - 2))
        y0 = max(0, min(h - 1, c.y - 2))
        x1 = max(x0 + 1, min(w, c.x + c.width + 2))
        y1 = max(y0 + 1, min(h, c.y + c.height + 2))
        radius = min(3, max(1, (x1 - x0) // 2, (y1 - y0) // 2))
        draw_3c.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=(99, 102, 241), width=1)

    # 4. Generate Bounding Boxes & Detailed Cards with Thumbnails
    bounding_boxes = []
    diff_details = []

    for c in changes:
        bounding_boxes.append({
            'id': c.id,
            'x': c.x,
            'y': c.y,
            'width': c.width,
            'height': c.height,
            'change_type': c.change_type,
        })

        pad = 6
        crop_x0 = max(0, min(w, c.x - pad))
        crop_y0 = max(0, min(h, c.y - pad))
        crop_x1 = max(crop_x0 + 1, min(w, c.x + c.width + pad))
        crop_y1 = max(crop_y0 + 1, min(h, c.y + c.height + pad))

        # Generate snippets for the top 100 differences
        if c.id <= 100 and crop_x1 > crop_x0 and crop_y1 > crop_y0:
            thumb_a = image_to_b64(img_a_clean.crop((crop_x0, crop_y0, crop_x1, crop_y1)))
            thumb_b = image_to_b64(img_b_clean.crop((crop_x0, crop_y0, crop_x1, crop_y1)))
        else:
            thumb_a = ''
            thumb_b = ''

        if c.change_type == 'removed':
            label = 'Removed from Document A'
        elif c.change_type == 'added':
            label = 'Added in Document B'
        else:
            label = 'Content Modified'

        diff_details.append({
            'id': c.id,
            'x': c.x,
            'y': c.y,
            'width': c.width,
            'height': c.height,
            'change_type': c.change_type,
            'label': label,
            'description': c.description,
            'thumb_a_b64': thumb_a,
            'thumb_b_b64': thumb_b,
        })

    # Pixel diff statistics
    total_pixels = w * h
    stats_mask = np.zeros((h, w), dtype=bool)
    for c in changes:
        cx0 = max(0, min(w, c.x))
        cy0 = max(0, min(h, c.y))
        cx1 = max(cx0, min(w, c.x + c.width))
        cy1 = max(cy0, min(h, c.y + c.height))
        if cx1 > cx0 and cy1 > cy0:
            stats_mask[cy0:cy1, cx0:cx1] = True
    if changed_pixels_mask is not None:
        stats_mask |= changed_pixels_mask

    diff_pixels = int(stats_mask.sum())
    changed_pct = round((diff_pixels / max(1, total_pixels)) * 100.0, 2)
    same_pct = round(max(0.0, 100.0 - changed_pct), 2)

    return {
        'img_a_b64': image_to_b64(img_a_clean),
        'img_b_b64': image_to_b64(img_b_clean),
        'img_a_highlighted_b64': image_to_b64(img_a_highlighted),
        'img_b_highlighted_b64': image_to_b64(img_b_highlighted),
        'diff_image_b64': image_to_b64(diff_img),
        'diff_3c_b64': image_to_b64(diff_3c_img),
        'bounding_boxes': bounding_boxes,
        'diff_details': diff_details,
        'differences_count': len(changes),
        'changed_pct': changed_pct,
        'same_pct': same_pct,
        'diff_pixels': diff_pixels,
        'total_pixels': total_pixels,
        'width': w,
        'height': h,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Line-Level Text Difflib for Inspector
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_diff(bytes_a: bytes, bytes_b: bytes, page_num: int = 1) -> list[dict]:
    """
    Extract line-level text differences for inspector view.
    Only operates on actual PDF files; returns [] silently for images.
    Never raises — all exceptions are swallowed so they never cause a 500.
    """
    if not _HAS_FITZ:
        return []

    def _quick_is_pdf(b: bytes) -> bool:
        return b'%PDF-' in b[:8192]

    if not _quick_is_pdf(bytes_a) or not _quick_is_pdf(bytes_b):
        return []

    doc_a = doc_b = None
    try:
        doc_a = _open_pdf(bytes_a)
        doc_b = _open_pdf(bytes_b)

        idx_a = max(0, min(page_num - 1, doc_a.page_count - 1))
        idx_b = max(0, min(page_num - 1, doc_b.page_count - 1))

        t_a = doc_a[idx_a].get_text('text') if doc_a.page_count > 0 else ''
        t_b = doc_b[idx_b].get_text('text') if doc_b.page_count > 0 else ''

        lines_a = [ln.strip() for ln in t_a.splitlines() if ln.strip()]
        lines_b = [ln.strip() for ln in t_b.splitlines() if ln.strip()]

        if not lines_a and not lines_b:
            return []

        matcher = difflib.SequenceMatcher(None, lines_a, lines_b, autojunk=False)
        text_changes = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            orig = lines_a[i1:i2]
            rev  = lines_b[j1:j2]
            text_changes.append({
                'type': 'modified' if tag == 'replace' else ('deleted' if tag == 'delete' else 'added'),
                'original': ' '.join(orig),
                'revised':  ' '.join(rev),
            })
        return text_changes

    except Exception:
        logger.warning("Text diff extraction skipped (non-critical)", exc_info=True)
        return []
    finally:
        for _d in (doc_a, doc_b):
            if _d is not None:
                try:
                    if not _d.is_closed:
                        _d.close()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Main Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def compare_documents(bytes_a: bytes, filename_a: str,
                      bytes_b: bytes, filename_b: str,
                      page: int = 1, sensitivity: int = 10,
                      max_dim: int = 1400) -> dict[str, Any]:
    """
    Public entry point to compare two documents (PDF or image).
    Seamlessly handles PDF vs PDF, Image vs Image, and mixed PDF vs Image comparisons.
    Returns an /api/compare-compatible dictionary.
    """
    sensitivity = max(1, min(50, int(sensitivity)))
    page = max(1, int(page))

    def _looks_like_pdf(name: str, payload: bytes) -> bool:
        if (name or '').lower().endswith('.pdf'):
            return True
        return b'%PDF-' in payload[:4096]

    is_pdf_a = _looks_like_pdf(filename_a, bytes_a)
    is_pdf_b = _looks_like_pdf(filename_b, bytes_b)
    both_are_pdfs = is_pdf_a and is_pdf_b and _HAS_FITZ

    pages_a = 1
    pages_b = 1
    text_diff: list[dict] = []
    matched_word_count = 0

    # Pre-initialize all variables used after the if/else block.
    # This prevents UnboundLocalError if an exception is raised partway
    # through the both_are_pdfs try block (e.g. during page rendering),
    # since the finally clause still runs but the variables would be unbound.
    arr_a: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
    arr_b: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
    all_changes: list[Change] = []
    changed_pixels_mask: np.ndarray = np.zeros((1, 1), dtype=bool)

    if both_are_pdfs:
        doc_a = None
        doc_b = None
        try:
            doc_a = _open_pdf(bytes_a)
            doc_b = _open_pdf(bytes_b)
            pages_a = max(1, doc_a.page_count)
            pages_b = max(1, doc_b.page_count)

            idx_a = max(0, min(page - 1, pages_a - 1))
            idx_b = max(0, min(page - 1, pages_b - 1))

            page_a = doc_a.load_page(idx_a)
            page_b = doc_b.load_page(idx_b)

            max_pt_w = max(float(page_a.rect.width or 0), float(page_b.rect.width or 0), 1.0)
            max_pt_h = max(float(page_a.rect.height or 0), float(page_b.rect.height or 0), 1.0)
            scale = min(2.0, float(max_dim) / max(max_pt_w, max_pt_h))
            scale = max(0.05, scale)

            try:
                arr_a_raw = _render_pdf_page(page_a, scale)
                arr_b_raw = _render_pdf_page(page_b, scale)
            except Exception as exc:
                raise ValueError(
                    'A PDF page could not be rendered. The file may be damaged or use an unsupported color space.'
                ) from exc

            arr_a, arr_b = _pad_bottom_right(arr_a_raw, arr_b_raw)
            H, W = arr_a.shape[:2]

            all_changes = []
            handled_mask = np.zeros((H, W), dtype=bool)

            # 1. Text comparison (pinpoint word boxes)
            text_changes, mask_text, matched_word_count = _compare_text_words(
                page_a, page_b, scale, W, H
            )
            all_changes.extend(text_changes)
            handled_mask |= mask_text

            # 2. Diagram / drawing comparison
            draw_changes, mask_draw = _compare_drawings(
                page_a, page_b, arr_a, arr_b, scale, sensitivity, W, H
            )
            all_changes.extend(draw_changes)
            handled_mask |= mask_draw

            # 3. Embedded image comparison
            img_changes, mask_img = _compare_images(
                page_a, page_b, arr_a, arr_b, scale, sensitivity, W, H
            )
            all_changes.extend(img_changes)
            handled_mask |= mask_img

            # 4. Residual pixel pass
            pixel_changes, changed_pixels_mask = _residual_pixel_diff(
                arr_a, arr_b, handled_mask, sensitivity
            )
            all_changes.extend(pixel_changes)

            # Line-level difflib for inspector
            text_diff = extract_text_diff(bytes_a, bytes_b, page)

        finally:
            for _doc in (doc_a, doc_b):
                try:
                    if _doc is not None and not _doc.is_closed:
                        _doc.close()
                except Exception:
                    pass

    else:
        # Non-PDF or mixed PDF-Image comparison
        img_a_pil, pages_a = _load_image(bytes_a, filename_a, page, max_dim)
        img_b_pil, pages_b = _load_image(bytes_b, filename_b, page, max_dim)

        # Ensure both are white-background RGB (no alpha channel)
        img_a_pil = _convert_to_white_bg_rgb(img_a_pil)
        img_b_pil = _convert_to_white_bg_rgb(img_b_pil)

        ar_a = img_a_pil.width / max(1, img_a_pil.height)
        ar_b = img_b_pil.width / max(1, img_b_pil.height)

        # If aspect ratios match within ~2%, scale both to same size before diff
        if abs(ar_a - ar_b) / max(ar_a, ar_b) <= 0.02:
            target_w = max(img_a_pil.width, img_b_pil.width)
            target_h = max(img_a_pil.height, img_b_pil.height)
            if max(target_w, target_h) > max_dim:
                scale_f = max_dim / max(target_w, target_h)
                target_w = max(1, int(target_w * scale_f))
                target_h = max(1, int(target_h * scale_f))
            img_a_pil = img_a_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_b_pil = img_b_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            arr_a = np.array(img_a_pil, dtype=np.uint8)
            arr_b = np.array(img_b_pil, dtype=np.uint8)
        else:
            # Different aspect ratios: downscale if exceeding max_dim, then pad bottom/right with white
            if max(img_a_pil.width, img_a_pil.height) > max_dim:
                sa = max_dim / max(img_a_pil.width, img_a_pil.height)
                img_a_pil = img_a_pil.resize((max(1, int(img_a_pil.width * sa)), max(1, int(img_a_pil.height * sa))), Image.Resampling.LANCZOS)
            if max(img_b_pil.width, img_b_pil.height) > max_dim:
                sb = max_dim / max(img_b_pil.width, img_b_pil.height)
                img_b_pil = img_b_pil.resize((max(1, int(img_b_pil.width * sb)), max(1, int(img_b_pil.height * sb))), Image.Resampling.LANCZOS)
            arr_a_raw = np.array(img_a_pil, dtype=np.uint8)
            arr_b_raw = np.array(img_b_pil, dtype=np.uint8)
            arr_a, arr_b = _pad_bottom_right(arr_a_raw, arr_b_raw)

        H, W = arr_a.shape[:2]
        handled_mask = np.zeros((H, W), dtype=bool)

        pixel_changes, changed_pixels_mask = _residual_pixel_diff(
            arr_a, arr_b, handled_mask, sensitivity
        )
        all_changes = pixel_changes
        matched_word_count = 0

        # If both files happen to be PDFs, still extract text difflib
        if is_pdf_a and is_pdf_b:
            text_diff = extract_text_diff(bytes_a, bytes_b, page)

    # Ensure arrays are always 3-channel RGB before building PIL images
    def _to_rgb3(a: np.ndarray) -> np.ndarray:
        if a.ndim == 2:
            return np.stack([a, a, a], axis=2)
        if a.shape[2] == 4:
            return a[:, :, :3]
        return a

    arr_a = _to_rgb3(arr_a)
    arr_b = _to_rgb3(arr_b)

    img_a_clean = Image.fromarray(arr_a, 'RGB')
    img_b_clean = Image.fromarray(arr_b, 'RGB')

    # Visual overlays & stats
    res = _generate_visuals(
        img_a_clean, img_b_clean, arr_a, arr_b, all_changes, changed_pixels_mask
    )

    # 5. Completely Different guard
    # When documents are completely unlike, suppress per-element bounding boxes
    # (too many / meaningless) but retain pixel-level statistics.
    is_completely_different = False
    changed_pct = res['changed_pct']

    if both_are_pdfs:
        if changed_pct >= 60.0 and matched_word_count == 0:
            is_completely_different = True
    else:
        if changed_pct >= 65.0:
            fg_a = arr_a.mean(axis=2) < 240
            fg_b = arr_b.mean(axis=2) < 240
            union = (fg_a | fg_b).sum()
            inter = (fg_a & fg_b).sum()
            iou = inter / max(1, union)
            if iou < 0.05:
                is_completely_different = True

    if is_completely_different:
        res['diff_details'] = []
        res['bounding_boxes'] = []
        res['differences_count'] = 0
        res['img_a_highlighted_b64'] = res['img_a_b64']
        res['img_b_highlighted_b64'] = res['img_b_b64']

    res['success'] = True
    res['is_completely_different'] = is_completely_different
    res['diff_b64'] = res['diff_image_b64']
    res['text_diff'] = text_diff
    res['current_page'] = page
    res['total_pages_a'] = pages_a
    res['total_pages_b'] = pages_b
    res['total_pages'] = max(pages_a, pages_b)

    return res


if __name__ == '__main__':
    pdf1 = r"SameSize/6866VBE92991338060W001 SH26 1 (1).pdf"
    pdf2 = r"SameSize/6866VBE92991338062W001 SH43 1 (1).pdf"
    if os.path.exists(pdf1) and os.path.exists(pdf2):
        compare_pdfs_overlay(pdf1, pdf2)
    else:
        print("DiffView difference engine initialized.")

