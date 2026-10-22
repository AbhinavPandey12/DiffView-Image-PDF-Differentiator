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
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage

# PyMuPDF for fast PDF rendering
import pymupdf as fitz

app = Flask(__name__)
# Allow uploads up to 128 MB
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024


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


def normalize_images(img_a: Image.Image, img_b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Pad both images to the exact same dimension (max W x max H) with a clean white background.
    Guarantees pixel-by-pixel spatial alignment for comparison.
    """
    w = max(img_a.width, img_b.width)
    h = max(img_a.height, img_b.height)

    def pad(img):
        if img.size == (w, h):
            return img
        canvas = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        canvas.paste(img, (0, 0))
        return canvas

    return pad(img_a), pad(img_b)


def compute_pixel_diff(img_a: Image.Image, img_b: Image.Image, sensitivity: int = 10) -> dict:
    """
    Vectorized NumPy & SciPy pixel difference engine.
    Computes:
      1. Precise pixel change mask.
      2. Morphologically grouped difference clusters and bounding boxes.
      3. Opaque white-background high-contrast Heatmap Diff with red overlays and bounding boxes.
      4. 3-Color Diff (Red = Removed from A, Green = Added in B, Amber = Modified).
      5. Side-by-Side highlighted images: Document A with Red highlights and Document B with Green highlights.
    """
    img_a, img_b = normalize_images(img_a, img_b)
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

        # ── Phase 2: cluster-level displacement suppression (up to 12 px) ───────
        # After micro-jitter filtering, surviving clusters are tested as whole
        # blobs.  For each cluster, the tight crop from doc A is slid across doc B
        # within a larger radius.  If the entire blob finds a near-identical match
        # (mean pixel diff ≤ sensitivity × 1.5) anywhere in that window the cluster
        # is only a positional/orientation shift of the same element and is
        # suppressed.  Genuine content changes (different shape, colour, text edit,
        # structural modification) will NOT find a good match and are preserved.
        MAX_CLUSTER_SHIFT = 12   # pixel radius for whole-cluster displacement test
        MATCH_THRESHOLD_SCALE = 1.5  # tolerance multiplier relative to sensitivity

        # Grow the phase-1 mask to identify surviving clusters
        dilated_p1 = ndimage.binary_dilation(phase1_mask, structure=struct, iterations=3)
        cluster_labels, _ = ndimage.label(dilated_p1)
        cluster_slices = ndimage.find_objects(cluster_labels)

        changed_mask = np.zeros((h, w), dtype=bool)

        for sl in cluster_slices:
            # Tight bounding box of this cluster in doc A
            cy0, cy1 = sl[0].start, sl[0].stop
            cx0, cx1 = sl[1].start, sl[1].stop
            crop_h = cy1 - cy0
            crop_w = cx1 - cx0

            # Skip trivially small clusters (handled by the area filter later)
            if crop_h * crop_w < 9:
                continue

            template = arr_a[cy0:cy1, cx0:cx1]  # shape (crop_h, crop_w, 3)

            # Search window in doc B — expand by MAX_CLUSTER_SHIFT on all sides
            wy0 = max(0, cy0 - MAX_CLUSTER_SHIFT)
            wy1 = min(h, cy1 + MAX_CLUSTER_SHIFT)
            wx0 = max(0, cx0 - MAX_CLUSTER_SHIFT)
            wx1 = min(w, cx1 + MAX_CLUSTER_SHIFT)

            window_b = arr_b[wy0:wy1, wx0:wx1]
            win_h, win_w, _ = window_b.shape

            best_mean_diff = float('inf')

            # Slide the template across the search window
            for dy in range(win_h - crop_h + 1):
                for dx in range(win_w - crop_w + 1):
                    patch = window_b[dy:dy + crop_h, dx:dx + crop_w]
                    mean_diff = float(np.abs(template - patch).mean())
                    if mean_diff < best_mean_diff:
                        best_mean_diff = mean_diff
                    # Early exit: already a very good match
                    if best_mean_diff <= sensitivity * 0.5:
                        break
                if best_mean_diff <= sensitivity * 0.5:
                    break

            # If the best match is within threshold → positional shift → suppress
            if best_mean_diff <= sensitivity * MATCH_THRESHOLD_SCALE:
                # Do not add this cluster to changed_mask (suppressed)
                continue

            # Genuine content difference — keep the phase-1 pixels for this cluster
            changed_mask[cy0:cy1, cx0:cx1] |= phase1_mask[cy0:cy1, cx0:cx1]

    diff_pixels = int(changed_mask.sum())
    total_pixels = w * h

    # Group difference pixels into readable word/block clusters using binary dilation
    struct = ndimage.generate_binary_structure(2, 2)
    dilated_mask = ndimage.binary_dilation(changed_mask, structure=struct, iterations=3)

    # Detect connected difference regions
    labeled_array, _ = ndimage.label(dilated_mask)
    objects = ndimage.find_objects(labeled_array)
    bboxes = []

    for obj in objects:
        ymin, ymax = obj[0].start, obj[0].stop
        xmin, xmax = obj[1].start, obj[1].stop
        bw = xmax - xmin
        bh = ymax - ymin
        # Filter tiny 1-2px isolated anti-aliasing artifacts
        if bw * bh >= 9:
            bboxes.append({
                'id': len(bboxes) + 1,
                'x': int(xmin),
                'y': int(ymin),
                'width': int(bw),
                'height': int(bh)
            })

    # Sort bounding boxes in reading order: top-to-bottom, left-to-right
    bboxes.sort(key=lambda b: (b['y'] // 30, b['x']))
    for i, b in enumerate(bboxes):
        b['id'] = i + 1

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

    # Document B with emerald green highlights on added/altered parts
    ov_b = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw_ob = ImageDraw.Draw(ov_b)
    for b in bboxes:
        draw_ob.rounded_rectangle(
            [b['x'] - 3, b['y'] - 3, b['x'] + b['width'] + 3, b['y'] + b['height'] + 3],
            radius=4, fill=(16, 185, 129, 50), outline=(16, 185, 129, 220), width=2
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
        bytes_a = file_a.read()
        bytes_b = file_b.read()

        if not bytes_a or not bytes_b:
            return jsonify({'error': 'One or both uploaded files are empty.'}), 400

        img_a, pages_a = file_to_image(bytes_a, file_a.filename, page)
        img_b, pages_b = file_to_image(bytes_b, file_b.filename, page)

        diff_result = compute_pixel_diff(img_a, img_b, sensitivity)
        img_a_norm, img_b_norm = normalize_images(img_a, img_b)

        # Extract text differences if files are PDFs
        text_diff = extract_text_diff(bytes_a, bytes_b, page)

        return jsonify({
            'success': True,
            'img_a_b64': image_to_b64(img_a_norm),
            'img_b_b64': image_to_b64(img_b_norm),
            'img_a_highlighted_b64': diff_result['img_a_highlighted_b64'],
            'img_b_highlighted_b64': diff_result['img_b_highlighted_b64'],
            'diff_b64': diff_result['diff_image_b64'],
            'diff_3c_b64': diff_result['diff_3c_b64'],
            'bounding_boxes': diff_result['bounding_boxes'],
            'diff_details': diff_result['diff_details'],
            'text_diff': text_diff,
            'differences_count': diff_result['differences_count'],
            'changed_pct': diff_result['changed_pct'],
            'same_pct': diff_result['same_pct'],
            'diff_pixels': diff_result['diff_pixels'],
            'total_pixels': diff_result['total_pixels'],
            'width': diff_result['width'],
            'height': diff_result['height'],
            'current_page': page,
            'total_pages_a': pages_a,
            'total_pages_b': pages_b,
            'total_pages': max(pages_a, pages_b),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to process comparison: {str(exc)}'}), 500


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

