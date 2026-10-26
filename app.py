"""
DiffView — PDF & Image Comparator
Flask Backend with PyMuPDF, Pillow, and NumPy

Spot visual differences between two PDF documents or image files side-by-side,
with pixel-accurate heatmap highlighting, 3-color add/remove diffing,
and split-screen comparison.
"""

import os
import io
import base64
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage
from werkzeug.exceptions import HTTPException

# PyMuPDF for fast PDF rendering
import pymupdf as fitz
import diff_engine
from diff_engine import compare_pdfs_overlay, pdf_to_image

app = Flask(__name__)
# Allow uploads up to 128 MB
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.config['MAX_DIFF_DIMENSION'] = int(os.environ.get('DIFFVIEW_MAX_DIMENSION', '1400'))
# Always use JSON error handlers for /api/* instead of Werkzeug HTML tracebacks
# (debug=True would otherwise hide the 500 handler and break the fetch client).
app.config['PROPAGATE_EXCEPTIONS'] = False

ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
ALLOWED_IMAGE_FORMATS = {'PNG', 'JPEG', 'WEBP', 'BMP', 'GIF', 'TIFF'}
logger = logging.getLogger(__name__)


def validate_upload(file_storage) -> bytes:
    """Read and validate an upload without trusting its browser-provided MIME type."""
    filename = (file_storage.filename or '').strip()
    extension = os.path.splitext(filename)[1].lower()
    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise ValueError('Unsupported file type. Use PDF, PNG, JPG, WEBP, BMP, GIF, or TIFF.')

    payload = file_storage.read()
    if not payload:
        raise ValueError('One or both uploaded files are empty.')

    is_pdf = payload[:1024].lstrip().startswith(b'%PDF-') or extension == '.pdf'
    if is_pdf:
        if b'%PDF-' not in payload[:4096]:
            raise ValueError('The uploaded PDF header is invalid or unreadable.')
        return payload

    try:
        with Image.open(io.BytesIO(payload)) as probe:
            if probe.format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError('The uploaded image format is not supported.')
            probe.verify()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('The uploaded image is invalid or unreadable.') from exc
    return payload


# ─────────────────────────────────────────────────────────────
# CORE IMAGE & PDF PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def file_to_image(file_bytes: bytes, filename: str, page: int = 1, scale: float = 1.5) -> tuple[Image.Image, int]:
    """
    Convert a PDF file page or image bytes to a Pillow RGBA Image.
    Returns (Pillow Image, total_pages_count).
    """
    ext = os.path.splitext(filename)[1].lower() if filename else ''
    is_pdf = ext == '.pdf' or filename.lower().endswith('.pdf') or (b'%PDF-' in file_bytes[:1024])

    if is_pdf:
        try:
            return pdf_page_to_image(file_bytes, page, scale=scale)
        except Exception as pdf_err:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                img = ImageOps.exif_transpose(img)
                return img.convert('RGBA'), 1
            except Exception:
                raise pdf_err
    else:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)
            return img.convert('RGBA'), 1
        except Exception:
            return pdf_page_to_image(file_bytes, page, scale=scale)


def pdf_page_to_image(file_bytes: bytes, page_num: int = 1, scale: float = 1.5) -> tuple[Image.Image, int]:
    """
    Render a single page of a PDF document into a Pillow Image at the given resolution scale.
    Handles stream sanitization, password authentication, and zero-page edge cases safely.
    Returns (Pillow Image, total_pages_count).
    """
    doc = None
    try:
        # Strip leading whitespace or metadata before %PDF- header if present
        pdf_start = file_bytes.find(b'%PDF-')
        clean_bytes = file_bytes[pdf_start:] if pdf_start > 0 else file_bytes

        try:
            doc = fitz.open(stream=clean_bytes, filetype='pdf')
        except Exception:
            doc = fitz.open(stream=file_bytes)

        if doc.needs_pass:
            doc.authenticate('')

        total_pages = doc.page_count

        if total_pages == 0:
            try:
                if doc is not None and not doc.is_closed:
                    doc.close()
            except Exception:
                pass
            doc = fitz.open(stream=file_bytes)
            total_pages = doc.page_count

        if total_pages == 0:
            try:
                if doc is not None and not doc.is_closed:
                    doc.close()
            except Exception:
                pass
            try:
                img = Image.open(io.BytesIO(file_bytes))
                img = ImageOps.exif_transpose(img)
                return img.convert('RGBA'), 1
            except Exception:
                raise ValueError(
                    "The PDF document contains 0 readable pages or could not be decoded. "
                    "Please ensure the file is a valid PDF and is not password-protected."
                )

        target_page = 1 if page_num < 1 else page_num
        page_idx = max(0, min(target_page - 1, total_pages - 1))
        page = doc.load_page(page_idx)

        # Render page with anti-aliasing
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples).convert('RGBA')
        return img, total_pages

    finally:
        try:
            if doc is not None and not doc.is_closed:
                doc.close()
        except Exception:
            pass


def normalize_images(img_a: Image.Image, img_b: Image.Image, max_dim: int = 1400) -> tuple[Image.Image, Image.Image]:
    """
    Constrain maximum dimension to max_dim (preserving aspect ratio) to guarantee fast
    execution and avoid memory spikes on cloud instances.
    Pad both images to the exact same dimension (max W x max H) with a clean white background.
    Guarantees pixel-by-pixel spatial alignment for comparison.
    """
    def constrain(img):
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img

    img_a = constrain(img_a)
    img_b = constrain(img_b)

    w = max(img_a.width, img_b.width)
    h = max(img_a.height, img_b.height)

    def pad(img):
        canvas = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        canvas.paste(img, (0, 0))
        return canvas

    return pad(img_a), pad(img_b)


def is_exact_same_visual_element(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    max_shift: int = 20,
    sensitivity: int = 10
) -> bool:
    """
    Determines if the visual content within/around a candidate bounding box (bx, by, bw, bh)
    in Document A and Document B is actually the exact same visual diagram/element,
    allowing for sub-pixel anti-aliasing, rasterization variations, stroke-width drift,
    or positional displacements of up to max_shift pixels in any direction.

    Safety rules:
      - Protects text: Small elements (height <= 26px and width <= 80px) are only tested
        with micro-shift (<= 4px) so that text words/characters are NEVER falsely matched
        to other words on nearby lines.
      - Protects genuine modifications: Requires high morphological overlap (>= 90%)
        and comparable foreground area ratio (>= 78%) so that shape/connector changes
        are ALWAYS retained as differences.
      - Suppresses identical diagrams: If the diagram has the same structure and colors
        (even with 1-2px stroke variation, rasterization jitter, or small displacement),
        it is recognized as identical.
    """
    h, w, _ = arr_a.shape

    is_small_or_text = (bh <= 26 and bw <= 80)
    effective_shift = 4 if is_small_or_text else min(max_shift, 20)

    pad = 4
    y0 = max(0, by - pad)
    y1 = min(h, by + bh + pad)
    x0 = max(0, bx - pad)
    x1 = min(w, bx + bw + pad)

    crop_h = y1 - y0
    crop_w = x1 - x0
    if crop_h < 3 or crop_w < 3:
        return True

    crop_a = arr_a[y0:y1, x0:x1]
    crop_b = arr_b[y0:y1, x0:x1]

    # Foreground threshold (< 235 on any channel indicates ink / shape / text / diagram)
    fg_a = (crop_a < 235).any(axis=2)
    fg_b = (crop_b < 235).any(axis=2)
    count_a = int(fg_a.sum())
    count_b = int(fg_b.sum())

    # If virtually no foreground in either, treat as unchanged
    if count_a < 4 and count_b < 4:
        return True
    if (count_a < 4 and count_b >= 10) or (count_b < 4 and count_a >= 10):
        return False

    area_ratio = min(count_a, count_b) / max(count_a, count_b)
    if area_ratio < 0.78:
        return False

    struct_2 = ndimage.generate_binary_structure(2, 2)

    # ── Fast Path 1: Direct Position Check (tolerant of 1-2px stroke width & anti-aliasing) ──
    dilated_fg_b = ndimage.binary_dilation(fg_b, structure=struct_2, iterations=2)
    dilated_fg_a = ndimage.binary_dilation(fg_a, structure=struct_2, iterations=2)
    cov_a_in_b = float((fg_a & dilated_fg_b).sum()) / count_a
    cov_b_in_a = float((fg_b & dilated_fg_a).sum()) / count_b

    # At direct position, require 92%+ coverage in both directions
    if cov_a_in_b >= 0.92 and cov_b_in_a >= 0.92:
        fg_inter = fg_a & fg_b
        if fg_inter.sum() >= 4:
            mean_color_diff = float(np.abs(crop_a[fg_inter] - crop_b[fg_inter]).mean())
            if mean_color_diff <= max(22.0, sensitivity * 2.0):
                return True
        else:
            return True

    if is_small_or_text:
        return False

    # ── Path 2: Search for Shifted Match within effective_shift ──
    t_h, t_w, _ = crop_a.shape
    fg_rows, fg_cols = np.where(fg_a)
    n_pts = len(fg_rows)
    if n_pts == 0:
        return True

    if n_pts > 80:
        idx = np.linspace(0, n_pts - 1, 80, dtype=int)
        sample_r = fg_rows[idx]
        sample_c = fg_cols[idx]
    else:
        sample_r = fg_rows
        sample_c = fg_cols
    sample_fg = crop_a[sample_r, sample_c]

    wy0 = max(0, y0 - effective_shift)
    wy1 = min(h, y0 + t_h + effective_shift)
    wx0 = max(0, x0 - effective_shift)
    wx1 = min(w, x0 + t_w + effective_shift)

    window = arr_b[wy0:wy1, wx0:wx1]
    win_h, win_w, _ = window.shape
    max_dy = win_h - t_h
    max_dx = win_w - t_w
    if max_dy < 0 or max_dx < 0:
        return False

    best_score = float('inf')
    best_dy, best_dx = 0, 0

    step = 3 if max(max_dy, max_dx) > 8 else 1
    for dy in range(0, max_dy + 1, step):
        for dx in range(0, max_dx + 1, step):
            patch = window[dy + sample_r, dx + sample_c]
            score = float(np.abs(sample_fg - patch).mean())
            if score < best_score:
                best_score = score
                best_dy, best_dx = dy, dx
            if best_score <= 3.0:
                break
        if best_score <= 3.0:
            break

    if step > 1 and best_score > 3.0:
        for dy in range(max(0, best_dy - 2), min(max_dy, best_dy + 2) + 1):
            for dx in range(max(0, best_dx - 2), min(max_dx, best_dx + 2) + 1):
                patch = window[dy + sample_r, dx + sample_c]
                score = float(np.abs(sample_fg - patch).mean())
                if score < best_score:
                    best_score = score
                    best_dy, best_dx = dy, dx

    if best_score > max(24.0, sensitivity * 2.0):
        return False

    target_y = wy0 + best_dy
    target_x = wx0 + best_dx
    if target_y + t_h > h or target_x + t_w > w:
        return False

    target_crop = arr_b[target_y:target_y + t_h, target_x:target_x + t_w]
    fg_target = (target_crop < 235).any(axis=2)
    cnt_target = int(fg_target.sum())

    if cnt_target < 4:
        return False

    target_area_ratio = min(count_a, cnt_target) / max(count_a, cnt_target)
    if target_area_ratio < 0.80:
        return False

    dilated_target = ndimage.binary_dilation(fg_target, structure=struct_2, iterations=2)
    cov_src = float((fg_a & dilated_target).sum()) / count_a
    dilated_src = ndimage.binary_dilation(fg_a, structure=struct_2, iterations=2)
    cov_target = float((fg_target & dilated_src).sum()) / cnt_target

    if cov_src >= 0.90 and cov_target >= 0.90:
        inter = fg_a & fg_target
        if inter.sum() >= 4:
            col_diff = float(np.abs(crop_a[inter] - target_crop[inter]).mean())
            if col_diff <= max(25.0, sensitivity * 2.0):
                return True
        else:
            return True

    return False


def compute_pixel_diff(
    img_a: Image.Image,
    img_b: Image.Image,
    sensitivity: int = 10,
    max_dim: int = 1400,
) -> dict:
    """
    Vectorized NumPy & SciPy pixel difference engine.
    Computes:
      1. Precise pixel change mask.
      2. Morphologically grouped difference clusters and bounding boxes.
      3. Opaque white-background high-contrast Heatmap Diff with red overlays and bounding boxes.
      4. 3-Color Diff (Red = Removed from A, Green = Added in B, Amber = Modified).
      5. Side-by-Side highlighted images: Document A with Red highlights and Document B with Green highlights.
    """
    img_a, img_b = normalize_images(img_a, img_b, max_dim=max(256, int(max_dim)))
    w, h = img_a.size

    arr_a = np.array(img_a.convert('RGB'), dtype=np.float32)
    arr_b = np.array(img_b.convert('RGB'), dtype=np.float32)

    diff_0 = np.abs(arr_a - arr_b).mean(axis=2)
    raw_changed = diff_0 > sensitivity

    # If no candidate differences exist under raw direct comparison, exit early
    if not np.any(raw_changed):
        changed_mask = raw_changed
    else:
        # ── Phase 1: per-pixel micro-jitter suppression (unchanged, ≤5 px) ──────
        # Eliminates sub-pixel anti-aliasing noise, font kerning drift, and
        # PDF rasterisation jitter pixel-by-pixel using a circular neighbourhood.
        micro_shift = 5  # pixel radius for individual-pixel jitter
        struct = ndimage.generate_binary_structure(2, 2)
        dilated_raw = ndimage.binary_dilation(raw_changed, structure=struct, iterations=3)
        labeled, _ = ndimage.label(dilated_raw)
        slices = ndimage.find_objects(labeled)

        phase1_mask = np.zeros((h, w), dtype=bool)

        micro_shifts = [
            (dy, dx)
            for dy in range(-micro_shift, micro_shift + 1)
            for dx in range(-micro_shift, micro_shift + 1)
            if (dx != 0 or dy != 0) and (dx * dx + dy * dy <= micro_shift * micro_shift)
        ]

        for sl in slices:
            ymin = max(0, sl[0].start - micro_shift)
            ymax = min(h, sl[0].stop + micro_shift)
            xmin = max(0, sl[1].start - micro_shift)
            xmax = min(w, sl[1].stop + micro_shift)

            sub_a = arr_a[ymin:ymax, xmin:xmax]
            sub_b = arr_b[ymin:ymax, xmin:xmax]
            sub_h, sub_w, _ = sub_a.shape

            min_diff_a = np.abs(sub_a - sub_b).mean(axis=2)
            min_diff_b = min_diff_a.copy()

            for dy, dx in micro_shifts:
                y_a_start, y_a_end = max(0, -dy), min(sub_h, sub_h - dy)
                x_a_start, x_a_end = max(0, -dx), min(sub_w, sub_w - dx)
                y_b_start, y_b_end = max(0, dy), min(sub_h, sub_h + dy)
                x_b_start, x_b_end = max(0, dx), min(sub_w, sub_w + dx)

                d = np.abs(
                    sub_a[y_a_start:y_a_end, x_a_start:x_a_end] -
                    sub_b[y_b_start:y_b_end, x_b_start:x_b_end]
                ).mean(axis=2)

                cur_a = min_diff_a[y_a_start:y_a_end, x_a_start:x_a_end]
                np.minimum(cur_a, d, out=cur_a)

                cur_b = min_diff_b[y_b_start:y_b_end, x_b_start:x_b_end]
                np.minimum(cur_b, d, out=cur_b)

            sub_changed = (min_diff_a > sensitivity) | (min_diff_b > sensitivity)
            phase1_mask[ymin:ymax, xmin:xmax] |= sub_changed

        # ── Phase 2: cluster-level displacement suppression (up to 20 px) ───────
        # After micro-jitter filtering, surviving clusters are tested as whole
        # elements. If the element finds a matching appearance in the revised document
        # within max_shift pixels (or vice-versa), it is recognized as the same element
        # undergoing minor positional/orientation displacement and is suppressed.
        MAX_CLUSTER_SHIFT = 20   # pixel radius for whole-cluster displacement test

        # Grow the phase-1 mask to identify surviving clusters
        dilated_p1 = ndimage.binary_dilation(phase1_mask, structure=struct, iterations=3)
        cluster_labels, _ = ndimage.label(dilated_p1)
        cluster_slices = ndimage.find_objects(cluster_labels)

        changed_mask = np.zeros((h, w), dtype=bool)

        for sl in cluster_slices:
            cy0, cy1 = sl[0].start, sl[0].stop
            cx0, cx1 = sl[1].start, sl[1].stop
            crop_h = cy1 - cy0
            crop_w = cx1 - cx0

            if crop_h * crop_w < 9:
                continue

            # Check if this cluster represents an identical element that moved
            if is_exact_same_visual_element(arr_a, arr_b, cx0, cy0, crop_w, crop_h, max_shift=MAX_CLUSTER_SHIFT, sensitivity=sensitivity):
                continue

            # Genuine content difference — keep the phase-1 pixels for this cluster
            changed_mask[cy0:cy1, cx0:cx1] |= phase1_mask[cy0:cy1, cx0:cx1]

    # Group difference pixels into readable word/block clusters using binary dilation
    struct = ndimage.generate_binary_structure(2, 2)
    dilated_mask = ndimage.binary_dilation(changed_mask, structure=struct, iterations=3)

    # Detect connected difference regions
    labeled_array, _ = ndimage.label(dilated_mask)
    objects = ndimage.find_objects(labeled_array)
    raw_bboxes = []

    for obj in objects:
        ymin, ymax = obj[0].start, obj[0].stop
        xmin, xmax = obj[1].start, obj[1].stop
        bw = xmax - xmin
        bh = ymax - ymin
        # Filter tiny 1-2px isolated anti-aliasing artifacts
        if bw * bh >= 9:
            raw_bboxes.append({
                'x': int(xmin),
                'y': int(ymin),
                'width': int(bw),
                'height': int(bh)
            })

    # Post-cluster verification: ensure no bounding box contains visual diagrams
    # that are actually the exact same between Document A and Document B.
    bboxes = []
    for b in raw_bboxes:
        bx, by, bw, bh = b['x'], b['y'], b['width'], b['height']
        if is_exact_same_visual_element(arr_a, arr_b, bx, by, bw, bh, max_shift=20, sensitivity=sensitivity):
            # Suppress identical diagram: clear mask pixels so neither side-by-side
            # highlights nor the visual log shows it.
            changed_mask[by:by+bh, bx:bx+bw] = False
            dilated_mask[by:by+bh, bx:bx+bw] = False
            continue
        bboxes.append(b)

    # Sort bounding boxes in reading order: top-to-bottom, left-to-right
    bboxes.sort(key=lambda b: (b['y'] // 30, b['x']))
    for i, b in enumerate(bboxes):
        b['id'] = i + 1

    diff_pixels = int(changed_mask.sum())
    total_pixels = w * h

    # ─────────────────────────────────────────────────────────────
    # 1. HIGH-CONTRAST HEATMAP DIFF (Crisp white opaque paper background)
    # ─────────────────────────────────────────────────────────────
    diff_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    unchanged = ~dilated_mask

    # Preserve original text/structure in soft clean gray for context
    base_a = np.array(img_a.convert('RGB'), dtype=np.uint8)
    diff_canvas[unchanged] = np.clip(base_a[unchanged] * 0.75 + 58, 0, 255)

    # Light red translucent tint across dilated change cluster
    diff_canvas[dilated_mask] = [254, 226, 226]

    # Vivid red on the exact changed pixels
    diff_canvas[changed_mask] = [220, 38, 38]

    diff_img = Image.fromarray(diff_canvas, 'RGB')
    draw_diff = ImageDraw.Draw(diff_img)
    for b in bboxes:
        draw_diff.rounded_rectangle(
            [b['x'] - 2, b['y'] - 2, b['x'] + b['width'] + 2, b['y'] + b['height'] + 2],
            radius=3, outline=(220, 38, 38), width=2
        )

    # ─────────────────────────────────────────────────────────────
    # 2. 3-COLOR VISUAL DIFF (Removed vs Added vs Modified)
    # ─────────────────────────────────────────────────────────────
    diff_3c = np.full((h, w, 3), 255, dtype=np.uint8)
    diff_3c[unchanged] = np.clip(base_a[unchanged] * 0.75 + 58, 0, 255)

    val_a = arr_a.mean(axis=2)
    val_b = arr_b.mean(axis=2)
    # Removed from A: was dark in A, now white/light in B
    is_removed = changed_mask & (val_a < 240) & (val_b >= 215)
    # Added in B: was white in A, now dark in B
    is_added = changed_mask & (val_a >= 215) & (val_b < 240)
    # Modified: color or shape altered
    is_mod = changed_mask & ~(is_removed | is_added)

    diff_3c[dilated_mask] = [248, 250, 252]
    diff_3c[is_removed] = [239, 68, 68]   # Red for removed
    diff_3c[is_added] = [16, 185, 129]    # Green for added
    diff_3c[is_mod] = [245, 158, 11]      # Amber for modified

    diff_3c_img = Image.fromarray(diff_3c, 'RGB')
    draw_3c = ImageDraw.Draw(diff_3c_img)
    for b in bboxes:
        draw_3c.rounded_rectangle(
            [b['x'] - 2, b['y'] - 2, b['x'] + b['width'] + 2, b['y'] + b['height'] + 2],
            radius=3, outline=(99, 102, 241), width=1
        )

    # ─────────────────────────────────────────────────────────────
    # 3. SIDE-BY-SIDE HIGHLIGHTED OVERLAYS
    # ─────────────────────────────────────────────────────────────
    # Document A with red highlights on removed/altered parts
    ov_a = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw_oa = ImageDraw.Draw(ov_a)
    for b in bboxes:
        draw_oa.rounded_rectangle(
            [b['x'] - 3, b['y'] - 3, b['x'] + b['width'] + 3, b['y'] + b['height'] + 3],
            radius=4, fill=(239, 68, 68, 50), outline=(239, 68, 68, 220), width=2
        )
    img_a_highlighted = Image.alpha_composite(img_a, ov_a)

    # Document B with vivid blue highlights on added/altered parts
    ov_b = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw_ob = ImageDraw.Draw(ov_b)
    for b in bboxes:
        draw_ob.rounded_rectangle(
            [b['x'] - 3, b['y'] - 3, b['x'] + b['width'] + 3, b['y'] + b['height'] + 3],
            radius=4, fill=(37, 99, 235, 50), outline=(37, 99, 235, 220), width=2
        )
    img_b_highlighted = Image.alpha_composite(img_b, ov_b)

    # Generate written descriptions and thumbnail snippets for each difference
    diff_details = []
    for b in bboxes:
        bx, by, bw, bh = b['x'], b['y'], b['width'], b['height']
        # Expand snippet boundary slightly for visual context
        pad_px = 6
        crop_x1 = max(0, bx - pad_px)
        crop_y1 = max(0, by - pad_px)
        crop_x2 = min(w, bx + bw + pad_px)
        crop_y2 = min(h, by + bh + pad_px)

        crop_a = img_a.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_b = img_b.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        buf_a, buf_b = io.BytesIO(), io.BytesIO()
        crop_a.save(buf_a, format='PNG')
        crop_b.save(buf_b, format='PNG')
        thumb_a = base64.b64encode(buf_a.getvalue()).decode('utf-8')
        thumb_b = base64.b64encode(buf_b.getvalue()).decode('utf-8')

        # Determine if added, removed, or modified
        val_crop_a = arr_a[by:by+bh, bx:bx+bw].mean()
        val_crop_b = arr_b[by:by+bh, bx:bx+bw].mean()

        if val_crop_a < 235 and val_crop_b >= 215:
            change_type = 'removed'
            label = 'Removed from Document A'
            desc = f'Element previously located at (X: {bx}, Y: {by}, Size: {bw}×{bh}px) was removed in Document B.'
        elif val_crop_a >= 215 and val_crop_b < 235:
            change_type = 'added'
            label = 'Added in Document B'
            desc = f'New visual content (Size: {bw}×{bh}px) was added at (X: {bx}, Y: {by}) in Document B.'
        else:
            change_type = 'modified'
            label = 'Content Modified'
            desc = f'Visual content modified at (X: {bx}, Y: {by}, Size: {bw}×{bh}px). Discrepancy detected between versions.'

        diff_details.append({
            'id': b['id'],
            'x': bx,
            'y': by,
            'width': bw,
            'height': bh,
            'change_type': change_type,
            'label': label,
            'description': desc,
            'thumb_a_b64': thumb_a,
            'thumb_b_b64': thumb_b,
        })

    # Compute percentages
    changed_pct = (diff_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    same_pct = max(0.0, 100.0 - changed_pct)

    return {
        'diff_image_b64': image_to_b64(diff_img),
        'diff_3c_b64': image_to_b64(diff_3c_img),
        'img_a_highlighted_b64': image_to_b64(img_a_highlighted),
        'img_b_highlighted_b64': image_to_b64(img_b_highlighted),
        'bounding_boxes': bboxes,
        'diff_details': diff_details,
        'differences_count': len(bboxes),
        'changed_pct': round(changed_pct, 2),
        'same_pct': round(same_pct, 2),
        'diff_pixels': diff_pixels,
        'total_pixels': total_pixels,
        'width': w,
        'height': h,
    }


def extract_text_diff(bytes_a: bytes, bytes_b: bytes, page: int = 1) -> list:
    """
    Extract line-by-line and word-level text differences between two PDF documents.
    Returns a list of written text difference records.
    """
    import difflib
    try:
        pdf_start_a = bytes_a.find(b'%PDF-')
        clean_a = bytes_a[pdf_start_a:] if pdf_start_a >= 0 else bytes_a
        pdf_start_b = bytes_b.find(b'%PDF-')
        clean_b = bytes_b[pdf_start_b:] if pdf_start_b >= 0 else bytes_b

        doc_a = fitz.open(stream=clean_a, filetype='pdf')
        doc_b = fitz.open(stream=clean_b, filetype='pdf')

        if doc_a.needs_pass: doc_a.authenticate('')
        if doc_b.needs_pass: doc_b.authenticate('')

        idx_a = max(0, min(page - 1, doc_a.page_count - 1)) if doc_a.page_count > 0 else 0
        idx_b = max(0, min(page - 1, doc_b.page_count - 1)) if doc_b.page_count > 0 else 0

        t_a = doc_a[idx_a].get_text('text') if doc_a.page_count > 0 else ''
        t_b = doc_b[idx_b].get_text('text') if doc_b.page_count > 0 else ''

        try:
            if not doc_a.is_closed: doc_a.close()
            if not doc_b.is_closed: doc_b.close()
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
        return []


def image_to_b64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ─────────────────────────────────────────────────────────────
# APPLICATION ROUTES & API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main application interface."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'app': 'DiffView PDF & Image Comparator'})


@app.route('/api/compare', methods=['POST'])
def compare():
    """
    POST /api/compare
    Accepts multipart/form-data:
      - file_a: Original file (PDF or Image)
      - file_b: Revised file (PDF or Image)
      - page: Optional page number (1-indexed, default 1)
      - sensitivity: Optional tolerance (1-50, default 10)
    Returns JSON containing base64 images, highlighted views, and diff metrics.
    """
    if 'file_a' not in request.files or 'file_b' not in request.files:
        return jsonify({'error': 'Both file_a and file_b are required.'}), 400

    file_a = request.files['file_a']
    file_b = request.files['file_b']

    try:
        page = int(request.form.get('page', 1))
    except (ValueError, TypeError):
        page = 1

    try:
        sensitivity = int(request.form.get('sensitivity', 10))
    except (ValueError, TypeError):
        sensitivity = 10

    try:
        bytes_a = validate_upload(file_a)
        bytes_b = validate_upload(file_b)

        result = diff_engine.compare_documents(
            bytes_a=bytes_a,
            filename_a=file_a.filename or '',
            bytes_b=bytes_b,
            filename_b=file_b.filename or '',
            page=page,
            sensitivity=sensitivity,
            max_dim=app.config['MAX_DIFF_DIMENSION']
        )
        return jsonify(result)

    except MemoryError:
        logger.exception('Comparison ran out of memory')
        return jsonify({
            'error': 'The PDF is too complex or too large to compare in memory. Try a smaller page size or fewer vector objects.'
        }), 500
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('Comparison failed')
        return jsonify({
            'error': 'Failed to process the comparison. Check that both files are valid and try again.'
        }), 500


def _api_error_payload(message, status):
    return jsonify({'error': message}), status


@app.errorhandler(400)
def handle_bad_request(e):
    return _api_error_payload(str(getattr(e, 'description', e) or 'Bad request'), 400)


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith('/api/'):
        return _api_error_payload('Not found.', 404)
    return e


@app.errorhandler(413)
def handle_large_file(e):
    return _api_error_payload('Uploaded file exceeds the maximum size limit of 128MB.', 413)


@app.errorhandler(500)
def handle_500(e):
    return _api_error_payload('An internal server error occurred while processing the comparison.', 500)


@app.errorhandler(Exception)
def handle_uncaught(e):
    """Keep /api/* responses JSON even when Flask debug mode is on."""
    if isinstance(e, HTTPException):
        if request.path.startswith('/api/'):
            return _api_error_payload(str(e.description or e.name), e.code or 500)
        return e
    logger.exception('Unhandled server error')
    if request.path.startswith('/api/'):
        return _api_error_payload(
            'An internal server error occurred while processing the comparison.',
            500,
        )
    raise e


@app.route('/api/page', methods=['POST'])
def change_page():
    """
    POST /api/page
    Navigate to a specific page or recompute diff with new sensitivity.
    """
    return compare()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

# This block only matters when you run python app.py locally, and it works there as-is.
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('\n' + '=' * 60)
    print('  DiffView — PDF & Image Comparator (Python Server)')
    print(f'  Application running at: http://127.0.0.1:{port}')
    print('=' * 60 + '\n')
    app.run(debug=True, host='127.0.0.1', port=port)

