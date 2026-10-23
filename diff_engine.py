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
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
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


def _rect_to_pixels(rect: tuple[float, float, float, float], scale: float, max_w: int, max_h: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    """Convert PDF-points rect (x0, y0, x1, y1) to pixel bounds within (max_w, max_h)."""
    x0 = max(0, min(max_w, int(np.floor((rect[0] - pad) * scale))))
    y0 = max(0, min(max_h, int(np.floor((rect[1] - pad) * scale))))
    x1 = max(0, min(max_w, int(np.ceil((rect[2] + pad) * scale))))
    y1 = max(0, min(max_h, int(np.ceil((rect[3] + pad) * scale))))
    return (x0, y0, x1, y1)


def _crop_similar(arr_a: np.ndarray, arr_b: np.ndarray,
                  px0: int, py0: int, px1: int, py1: int, tol: float = 14.0) -> bool:
    """Mean absolute RGB delta within a bounding box; True -> visually identical."""
    h, w = arr_a.shape[:2]
    cx0 = max(0, px0)
    cy0 = max(0, py0)
    cx1 = min(w, px1)
    cy1 = min(h, py1)
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
            bx0 = max(0, px0 + dx)
            by0 = max(0, py0 + dy)
            bx1 = min(w, px1 + dx)
            by1 = min(h, py1 + dy)
            ax0 = max(0, px0)
            ay0 = max(0, py0)
            ax1 = min(w, px1)
            ay1 = min(h, py1)
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
    # Fallback to closest span
    best_dist = float('inf')
    best_style = ('', 0.0, 0)
    for sp in spans:
        sb = sp['bbox']
        scx = (sb[0] + sb[2]) / 2.0
        scy = (sb[1] + sb[3]) / 2.0
        dist = (cx - scx) ** 2 + (cy - scy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_style = (sp['font'], sp['size'], sp['color'])
    return best_style


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

    pw_a = page_a.rect.width or 1.0
    ph_a = page_a.rect.height or 1.0
    pw_b = page_b.rect.width or 1.0
    ph_b = page_b.rect.height or 1.0

    raw_words_a = page_a.get_text('words')
    raw_words_b = page_b.get_text('words')

    spans_a = _extract_spans(page_a)
    spans_b = _extract_spans(page_b)

    # Build word records
    def build_records(raw_words, pw, ph, spans):
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
                'cx_norm': cx / pw,
                'cy_norm': cy / ph,
                'matched': False,
            })
        return records

    words_a = build_records(raw_words_a, pw_a, ph_a, spans_a)
    words_b = build_records(raw_words_b, pw_b, ph_b, spans_b)

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

            px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0)
            px_b = _rect_to_pixels(best_b['rect'], scale, img_w, img_h, pad=1.0)

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

        # If found elsewhere on the page
        if best_b is not None and best_dist > POS_THRESH:
            wa['matched'] = True
            best_b['matched'] = True
            matched_word_count += 1

            px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0)
            px_b = _rect_to_pixels(best_b['rect'], scale, img_w, img_h, pad=1.0)

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

            px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0)
            px_b = _rect_to_pixels(best_b['rect'], scale, img_w, img_h, pad=1.0)

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
        px_a = _rect_to_pixels(wa['rect'], scale, img_w, img_h, pad=1.0)
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
        px_b = _rect_to_pixels(wb['rect'], scale, img_w, img_h, pad=1.0)
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
    items = d.get('items', [])
    color = d.get('color')
    fill = d.get('fill')
    width = d.get('width')
    def item_str(it):
        kind = it[0]
        if kind == 'l':
            return f"l({round(it[1][0],1)},{round(it[1][1],1)},{round(it[2][0],1)},{round(it[2][1],1)})"
        elif kind == 're':
            r = it[1]
            return f"re({round(r[0],1)},{round(r[1],1)},{round(r[2],1)},{round(r[3],1)})"
        elif kind == 'c':
            pts = it[1:]
            return 'c(' + ','.join(f"{round(p[0],1)},{round(p[1],1)}" for p in pts if hasattr(p, '__len__')) + ')'
        return str(kind)
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

    draw_a = page_a.get_drawings()
    draw_b = page_b.get_drawings()

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
        px_a = _rect_to_pixels(ra, scale, img_w, img_h, pad=stroke_pad_a)

        if best_iou >= 0.40 and best_bi >= 0:
            matched_b.add(best_bi)
            db = draw_b[best_bi]
            rb = get_rect(db)
            stroke_pad_b = max(1.5, float(db.get('width') or 1.0) / 2.0)
            px_b = _rect_to_pixels(rb, scale, img_w, img_h, pad=stroke_pad_b)

            sig_a = _drawing_sig(da)
            sig_b = _drawing_sig(db)
            is_same = (sig_a == sig_b)

            if not is_same:
                # Check shift-tolerant crop similarity in case coordinates slightly jittered
                is_same = _shift_tolerant_crop_similar(
                    arr_a, arr_b,
                    min(px_a[0], px_b[0]), min(px_a[1], px_b[1]),
                    max(px_a[2], px_b[2]), max(px_a[3], px_b[3]),
                    max_shift=4, tol=float(sensitivity) * 1.5,
                )

            # Mask both rects with generous padding to suppress line/edge AA noise
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
            # Unmatched drawing in A -> removed
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
        px_b = _rect_to_pixels(rb, scale, img_w, img_h, pad=stroke_pad_b)
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

    imgs_a = page_a.get_image_info(xrefs=True)
    imgs_b = page_b.get_image_info(xrefs=True)

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

        px_a = _rect_to_pixels(ra, scale, img_w, img_h)

        if best_iou >= 0.45 and best_bi >= 0:
            matched_b.add(best_bi)
            ib = imgs_b[best_bi]
            bbox_b = ib.get('bbox')
            rb = (float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3]))
            px_b = _rect_to_pixels(rb, scale, img_w, img_h)

            same = False
            # Check xref match or digest match
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
        px_b = _rect_to_pixels(rb, scale, img_w, img_h)
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
      • Tight min-rect bounding box around changed pixels
      • Shift check to eliminate shifted raster artifacts
    """
    changes: list[Change] = []
    h, w = arr_a.shape[:2]

    diff_raw = (np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16)).mean(axis=2) > sensitivity)
    diff_unhandled = diff_raw & (~handled_mask)

    # 1. binary_opening to drop 1px AA noise specks
    struct_3x3 = ndimage.generate_binary_structure(2, 2)
    opened = ndimage.binary_opening(diff_unhandled, structure=struct_3x3)

    # 2. Dilate ONCE (1x)
    dilated = ndimage.binary_dilation(opened, structure=struct_3x3, iterations=1)

    # 3. Connected components
    labeled, num_features = ndimage.label(dilated)
    if num_features == 0:
        return changes, opened

    for feat_id in range(1, num_features + 1):
        component_mask = (labeled == feat_id)
        # Exact changed pixels in this component
        active_pts = component_mask & opened
        if not np.any(active_pts):
            continue

        ys, xs = np.where(active_pts)
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        cw = max_x - min_x + 1
        ch = max_y - min_y + 1

        # Drop tiny specks (< 4 pixels)
        if len(xs) < 4:
            continue

        # Shift check to suppress shifted identical elements
        if _shift_tolerant_crop_similar(arr_a, arr_b, min_x, min_y, max_x + 1, max_y + 1,
                                         max_shift=4, tol=float(sensitivity) * 1.5):
            continue

        # Classify added, removed, or modified
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
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return arr.copy()


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
    clean = file_bytes[pdf_start:] if pdf_start > 0 else file_bytes
    try:
        doc = fitz.open(stream=clean, filetype='pdf')
    except Exception:
        doc = fitz.open(stream=file_bytes, filetype='pdf')
    if doc.needs_pass:
        doc.authenticate('')
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Visual Output Generation (Highlights, Heatmap, 3-Color, Snippets)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_visuals(img_a_clean: Image.Image, img_b_clean: Image.Image,
                      arr_a: np.ndarray, arr_b: np.ndarray,
                      changes: list[Change], changed_pixels_mask: np.ndarray) -> dict[str, Any]:
    """
    Generate all visual overlays adhering to the requested color specs:
      • Doc A: red = removed, amber = modified
      • Doc B: green = added, amber = modified
      • Heatmap: soft gray context + fill exact rects & residual changed pixels
      • 3-Color: red = removed, green = added, amber = modified
      • Thumbnail snippets for inspector cards
    """
    w, h = img_a_clean.size

    # Sort boxes in reading order: top-to-bottom, left-to-right
    changes.sort(key=lambda c: (c.y // 30, c.x))
    for i, c in enumerate(changes):
        c.id = i + 1

    # 1. Side-by-Side Highlighted Overlays
    ov_a = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov_b = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw_oa = ImageDraw.Draw(ov_a)
    draw_ob = ImageDraw.Draw(ov_b)

    for c in changes:
        x0 = max(0, c.x - 2)
        y0 = max(0, c.y - 2)
        x1 = min(w, c.x + c.width + 2)
        y1 = min(h, c.y + c.height + 2)

        if c.change_type == 'removed':
            # Document A: Red
            draw_oa.rounded_rectangle([x0, y0, x1, y1], radius=3,
                                      fill=(239, 68, 68, 55), outline=(239, 68, 68, 230), width=2)
        elif c.change_type == 'added':
            # Document B: Emerald Green
            draw_ob.rounded_rectangle([x0, y0, x1, y1], radius=3,
                                      fill=(16, 185, 129, 55), outline=(16, 185, 129, 230), width=2)
        else:
            # Modified: Amber in both documents
            draw_oa.rounded_rectangle([x0, y0, x1, y1], radius=3,
                                      fill=(245, 158, 11, 55), outline=(245, 158, 11, 230), width=2)
            draw_ob.rounded_rectangle([x0, y0, x1, y1], radius=3,
                                      fill=(245, 158, 11, 55), outline=(245, 158, 11, 230), width=2)

    img_a_highlighted = Image.alpha_composite(img_a_clean.convert('RGBA'), ov_a).convert('RGB')
    img_b_highlighted = Image.alpha_composite(img_b_clean.convert('RGBA'), ov_b).convert('RGB')

    # 2. High-Contrast Heatmap Diff
    diff_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    base_a = arr_a.copy()
    diff_canvas[:] = np.clip(base_a * 0.78 + 52, 0, 255)

    # Soft tint on exact change rects
    for c in changes:
        diff_canvas[c.y:c.y + c.height, c.x:c.x + c.width] = [254, 226, 226]

    # Vivid red on exact changed pixels
    if changed_pixels_mask is not None and np.any(changed_pixels_mask):
        diff_canvas[changed_pixels_mask] = [220, 38, 38]

    diff_img = Image.fromarray(diff_canvas, 'RGB')
    draw_diff = ImageDraw.Draw(diff_img)
    for c in changes:
        draw_diff.rounded_rectangle([c.x - 2, c.y - 2, c.x + c.width + 2, c.y + c.height + 2],
                                    radius=3, outline=(220, 38, 38), width=2)

    # 3. 3-Color Visual Diff (Removed vs Added vs Modified)
    diff_3c = np.full((h, w, 3), 255, dtype=np.uint8)
    diff_3c[:] = np.clip(base_a * 0.78 + 52, 0, 255)

    for c in changes:
        # Fill change rect according to change type
        if c.change_type == 'removed':
            diff_3c[c.y:c.y + c.height, c.x:c.x + c.width] = [239, 68, 68]
        elif c.change_type == 'added':
            diff_3c[c.y:c.y + c.height, c.x:c.x + c.width] = [16, 185, 129]
        else:
            diff_3c[c.y:c.y + c.height, c.x:c.x + c.width] = [245, 158, 11]

    diff_3c_img = Image.fromarray(diff_3c, 'RGB')
    draw_3c = ImageDraw.Draw(diff_3c_img)
    for c in changes:
        draw_3c.rounded_rectangle([c.x - 2, c.y - 2, c.x + c.width + 2, c.y + c.height + 2],
                                  radius=3, outline=(99, 102, 241), width=1)

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
        crop_x0 = max(0, c.x - pad)
        crop_y0 = max(0, c.y - pad)
        crop_x1 = min(w, c.x + c.width + pad)
        crop_y1 = min(h, c.y + c.height + pad)

        thumb_a = img_a_clean.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        thumb_b = img_b_clean.crop((crop_x0, crop_y0, crop_x1, crop_y1))

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
            'thumb_a_b64': image_to_b64(thumb_a),
            'thumb_b_b64': image_to_b64(thumb_b),
        })

    # Pixel diff statistics
    total_pixels = w * h
    # Union of bounding box areas + any residual changed pixels
    stats_mask = np.zeros((h, w), dtype=bool)
    for c in changes:
        stats_mask[c.y:c.y + c.height, c.x:c.x + c.width] = True
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
    """Extract line-level text differences for inspector view."""
    if not _HAS_FITZ:
        return []
    try:
        doc_a = _open_pdf(bytes_a)
        doc_b = _open_pdf(bytes_b)

        idx_a = max(0, min(page_num - 1, doc_a.page_count - 1)) if doc_a.page_count > 0 else 0
        idx_b = max(0, min(page_num - 1, doc_b.page_count - 1)) if doc_b.page_count > 0 else 0

        t_a = doc_a[idx_a].get_text('text') if doc_a.page_count > 0 else ''
        t_b = doc_b[idx_b].get_text('text') if doc_b.page_count > 0 else ''

        try:
            if not doc_a.is_closed:
                doc_a.close()
            if not doc_b.is_closed:
                doc_b.close()
        except Exception:
            pass

        lines_a = [l.strip() for l in t_a.splitlines() if l.strip()]
        lines_b = [l.strip() for l in t_b.splitlines() if l.strip()]

        if not lines_a and not lines_b:
            return []

        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
        text_changes = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            orig = lines_a[i1:i2]
            rev = lines_b[j1:j2]
            text_changes.append({
                'type': 'modified' if tag == 'replace' else ('deleted' if tag == 'delete' else 'added'),
                'original': ' '.join(orig),
                'revised': ' '.join(rev),
            })
        return text_changes
    except Exception:
        logger.exception("Text diff extraction failed")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def compare_documents(bytes_a: bytes, filename_a: str,
                      bytes_b: bytes, filename_b: str,
                      page: int = 1, sensitivity: int = 10,
                      max_dim: int = 1400) -> dict[str, Any]:
    """
    Public entry point to compare two documents (PDF or image).
    Returns an /api/compare-compatible dictionary.
    """
    sensitivity = max(1, min(50, int(sensitivity)))
    page = max(1, int(page))

    is_pdf_a = filename_a.lower().endswith('.pdf') or bytes_a[:4] == b'%PDF'
    is_pdf_b = filename_b.lower().endswith('.pdf') or bytes_b[:4] == b'%PDF'
    is_pdf = is_pdf_a and is_pdf_b and _HAS_FITZ

    pages_a = 1
    pages_b = 1
    text_diff = []

    if is_pdf:
        doc_a = _open_pdf(bytes_a)
        doc_b = _open_pdf(bytes_b)
        pages_a = doc_a.page_count
        pages_b = doc_b.page_count

        idx_a = max(0, min(page - 1, pages_a - 1))
        idx_b = max(0, min(page - 1, pages_b - 1))

        page_a = doc_a.load_page(idx_a)
        page_b = doc_b.load_page(idx_b)

        # 1. Align in PDF points: render both pages at the same scale
        max_pt_w = max(page_a.rect.width, page_b.rect.width, 1.0)
        max_pt_h = max(page_a.rect.height, page_b.rect.height, 1.0)
        scale = min(2.0, float(max_dim) / max(max_pt_w, max_pt_h))
        scale = max(0.5, scale)

        arr_a_raw = _render_pdf_page(page_a, scale)
        arr_b_raw = _render_pdf_page(page_b, scale)

        # Pad bottom/right only with white background
        arr_a, arr_b = _pad_bottom_right(arr_a_raw, arr_b_raw)
        H, W = arr_a.shape[:2]

        all_changes: list[Change] = []
        handled_mask = np.zeros((H, W), dtype=bool)

        # 2. Text comparison (pinpoint word boxes)
        text_changes, mask_text, matched_word_count = _compare_text_words(
            page_a, page_b, scale, W, H
        )
        all_changes.extend(text_changes)
        handled_mask |= mask_text

        # 3. Diagram / drawing comparison
        draw_changes, mask_draw = _compare_drawings(
            page_a, page_b, arr_a, arr_b, scale, sensitivity, W, H
        )
        all_changes.extend(draw_changes)
        handled_mask |= mask_draw

        # 4. Embedded image comparison
        img_changes, mask_img = _compare_images(
            page_a, page_b, arr_a, arr_b, scale, sensitivity, W, H
        )
        all_changes.extend(img_changes)
        handled_mask |= mask_img

        # 5. Residual pixel pass
        pixel_changes, changed_pixels_mask = _residual_pixel_diff(
            arr_a, arr_b, handled_mask, sensitivity
        )
        all_changes.extend(pixel_changes)

        # Line-level difflib for inspector
        text_diff = extract_text_diff(bytes_a, bytes_b, page)

        doc_a.close()
        doc_b.close()

    else:
        # Non-PDF or mixed image comparison
        img_a_in = Image.open(io.BytesIO(bytes_a)).convert('RGB')
        img_b_in = Image.open(io.BytesIO(bytes_b)).convert('RGB')

        ar_a = img_a_in.width / max(1, img_a_in.height)
        ar_b = img_b_in.width / max(1, img_b_in.height)

        # If aspect ratios match within ~2%, scale both to same size before diff
        if abs(ar_a - ar_b) / max(ar_a, ar_b) <= 0.02:
            target_w = max(img_a_in.width, img_b_in.width)
            target_h = max(img_a_in.height, img_b_in.height)
            if max(target_w, target_h) > max_dim:
                scale = max_dim / max(target_w, target_h)
                target_w = int(target_w * scale)
                target_h = int(target_h * scale)
            img_a_in = img_a_in.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_b_in = img_b_in.resize((target_w, target_h), Image.Resampling.LANCZOS)
            arr_a = np.array(img_a_in, dtype=np.uint8)
            arr_b = np.array(img_b_in, dtype=np.uint8)
        else:
            arr_a_raw = np.array(img_a_in, dtype=np.uint8)
            arr_b_raw = np.array(img_b_in, dtype=np.uint8)
            arr_a, arr_b = _pad_bottom_right(arr_a_raw, arr_b_raw)

        H, W = arr_a.shape[:2]
        handled_mask = np.zeros((H, W), dtype=bool)

        pixel_changes, changed_pixels_mask = _residual_pixel_diff(
            arr_a, arr_b, handled_mask, sensitivity
        )
        all_changes = pixel_changes
        matched_word_count = 0

    img_a_clean = Image.fromarray(arr_a, 'RGB')
    img_b_clean = Image.fromarray(arr_b, 'RGB')

    # Visual overlays & stats
    res = _generate_visuals(
        img_a_clean, img_b_clean, arr_a, arr_b, all_changes, changed_pixels_mask
    )

    # 6. Completely Different guard
    # Only when almost no matched words AND huge pixel disagreement
    is_completely_different = False
    changed_pct = res['changed_pct']

    if is_pdf:
        # Check if documents have zero textual overlap and high pixel delta
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
