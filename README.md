# DiffView — PDF & Image Visual & Textual Comparator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/PyMuPDF-High_Performance-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="PyMuPDF"/>
  <img src="https://img.shields.io/badge/NumPy-Vectorized_Diff-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/SciPy-Cluster_BBoxes-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/Pillow-Image_Processing-945DD6?style=for-the-badge" alt="Pillow"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

> **DiffView** is an open-source visual document and image comparison web application. It combines high-speed multi-stage PDF vector rasterization, 2D spatial grid indexing for dense CAD schematics, word-level text extraction with Levenshtein alignment, morphological computer vision clustering, and line-level text diffing to spot subtle modifications, design adjustments, typography edits, and visual alterations with pixel-level precision. Uploads are processed entirely in memory by the Flask service and are never persisted to disk.

---

## 📑 Table of Contents

- [Overview & Key Features](#-overview--key-features)
- [Exclusive Differences Breakdown & Written Analysis](#-exclusive-differences-breakdown--written-analysis)
- [Comparison Modes Breakdown](#-comparison-modes-breakdown)
- [How It Works (Technical Architecture)](#-how-it-works-technical-architecture)
- [Dense Engineering & CAD Drawing Optimization](#-dense-engineering--cad-drawing-optimization)
- [Directory Structure](#-directory-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [How to Run & Use the Application](#-how-to-run--use-the-application)
- [Keyboard Shortcuts](#-keyboard-shortcuts)
- [API Reference](#-api-reference)
- [Algorithm Deep-Dive](#-algorithm-deep-dive)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Contributing & License](#-contributing--license)

---

## ✨ Overview & Key Features

DiffView bridges the gap between raw document diffing and visual QA by analyzing pixel, vector, and textual differences across rendered PDF pages and bitmap images:

- 🎯 **Pixel-Level & Vector Precision**: Highlights altered elements with configurable color sensitivity tolerance ($1 - 50$) and pixel discrepancy statistics.
- 📝 **Differences Breakdown & Written Analysis**: A dedicated analytical component that breaks down and writes out all textual revisions and visual changes side-by-side with original vs. revised text snippets, thumbnail crops, and animated spotlights.
- ⚡ **Optimized for Engineering Schematics & CAD**: Features a 2D uniform spatial grid indexing engine in `diff_engine.py` that accelerates vector path comparisons on dense schematics (e.g. 9,200+ vector paths) from over 13 minutes down to under 3.5 seconds.
- 📑 **Full Multi-Page PDF Support**: Seamlessly render, cache, and navigate through multi-page PDFs using high-speed PyMuPDF rasterization.
- 🖼️ **Cross-Format Flexibility**: Compare `PDF`, `PNG`, `JPG`, `JPEG`, `WebP`, `BMP`, `GIF`, and `TIFF` files interchangeably (e.g. PDF vs. PDF, Image vs. Image, or PDF vs. Image).
- 🎛️ **4 Intuitive Comparison Modes**:
  1. **Side-by-Side**: Synchronized scrolling dual panes displaying Document A (with red bounding boxes) and Document B (with green bounding boxes).
  2. **Pixel Diff (Heatmap & 3-Color)**: High-contrast red heatmap overlay or 3-color breakdown (*Red = Removed, Green = Added, Amber = Modified*).
  3. **Slider Split Screen**: Interactive before-and-after drag-wipe slider overlay with adjustable opacity.
  4. **Flash / Flicker Mode**: Rapid A/B alternating flash on button hold or `Spacebar` to spot micro layout jitters and single-pixel shifts.
- 🎚️ **Dynamic Sensitivity Slider**: Tune the color distance tolerance on the fly to eliminate compression noise or capture microscopic sub-pixel shifts.
- 🔍 **Interactive Zoom & Pan**: Zoom smoothly from $25\%$ to $300\%$ with keyboard shortcuts and synchronized dual-pane panning.
- 💾 **Export Results**: Download high-resolution PNG diff visualizations directly to your computer.
- 🌓 **Dark / Light Mode**: Sleek dark interface with toggleable light mode persisted in `localStorage`.

---

## 🔎 Exclusive Differences Breakdown & Written Analysis

The application features a dedicated, comprehensive analytical panel (**#diff-inspector**) positioned directly below the document viewer:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Differences Breakdown & Written Analysis                                        │
│  [Total: 3]   [Removed: 1]   [Added: 1]   [Modified: 1]   [Text: 2]              │
├──────────────────────────────────────────────────────────────────────────────────┤
│  Written Text Comparison (Line-by-Line & Word-by-Word)                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ Diff #1  [MODIFIED]  Text Revision #1                                      │  │
│  │ - Original (A): Clause 1: Payment is due within 30 days.                   │  │
│  │ + Revised  (B): Clause 1: Payment is due within 15 days.                   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  Visual & Graphical Differences Log                                              │
│  ┌────────────────────────────────────────┐ ┌──────────────────────────────────┐ │
│  │ Diff #1  [MODIFIED]  [TEXT]            │ │ Diff #2  [REMOVED]  [DRAWING]    │ │
│  │ X: 118, Y: 450 (180×28px)              │ │ X: 640, Y: 320 (95×40px)         │ │
│  │ Text changed from "30 days" to "15...  │ │ Element removed in Document B    │ │
│  │ - Original: Clause 1: Payment is due...│ │ [No text - vector path]          │ │
│  │ + Revised:  Clause 1: Payment is due...│ │                                  │ │
│  │ ┌───────────────┐  ┌────────────────┐  │ │ ┌──────────────┐  ┌────────────┐ │ │
│  │ │  Doc A Crop   │  │   Doc B Crop   │  │ │ │  Doc A Crop  │  │ Doc B Crop │ │ │
│  │ └───────────────┘  └────────────────┘  │ │ └──────────────┘  └────────────┘ │ │
│  │ [Spotlight Button]                     │ │ [Spotlight Button]               │ │
│  └────────────────────────────────────────┘ └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Highlights of the Inspector:
1. **Synchronized Text & Visual Changes**:
   - Every detected change indicates its exact nature (`TEXT`, `DRAWING`, `IMAGE`, or `PIXEL`).
   - Cards that contain text modifications display both the original vs. revised text string comparison as well as high-resolution visual crop snippets of both documents.
2. **Line-by-Line & Word-by-Word Text Diff**:
   - Uses PyMuPDF text stream extraction, word bounding box indexing, and Python's `difflib.SequenceMatcher` to analyze line-level and word-level textual revisions.
   - Distinctly renders red original lines (`Original (A)`) and green revised lines (`Revised (B)`).
   - Clicking any text revision immediately spotlights its exact coordinate on the document viewer.
3. **Visual Differences Cards Grid**:
   - Every detected discrepancy cluster is assigned a card with bounding coordinates `(X, Y, W×H px)`.
   - Generates high-resolution cropped snippet thumbnails comparing Document A (red border) vs. Document B (blue border).
   - Detailed natural-language description explaining whether the element was removed, added, or modified.
4. **Animated Spotlight Navigation**:
   - Clicking **Spotlight** on any difference card triggers an animated pulsing spotlight box (`.diff-spotlight`) directly over the discrepancy on both documents and smoothly scrolls into view.
5. **Completely Different Notice**:
   - If two documents have completely mismatched content, dimensions, or aspect ratios (>90% different), a clear notice is presented to prevent unhelpful visual noise.

---

## 🎛️ Comparison Modes Breakdown

| Mode | Visual Mechanism | Best Used For |
| :--- | :--- | :--- |
| **Side-by-Side** | Dual synchronized panes with red (Doc A) and green (Doc B) bounding boxes & spotlight pulse | Layout comparison, typography checking, verifying margin shifts |
| **Pixel Diff (Heatmap)** | High-contrast white background with glowing red diff mask and bounding boxes | Identifying every single pixel discrepancy across complex documents |
| **Pixel Diff (3-Color)** | 3-Color classification: Red (Removed from A), Green (Added in B), Amber (Modified) | Auditing contract clauses, legal documents, engineering drawings |
| **Slider Split Screen** | Interactive before-and-after drag wipe with adjustable opacity slider | Validating alignment, stamp placement, watermark changes |
| **Flash / Flicker Mode** | Rapid A/B alternating flash on hold button or `Spacebar` | Spotting minute single-pixel shifts, subtle font weight discrepancies |
| **Differences Inspector** | Dedicated breakdown panel with written text diffs, visual cards & snippet crops | Item-by-item change verification and client audit reporting |

---

## 🛠️ How It Works (Technical Architecture)

DiffView utilizes a dual-engine architecture: a specialized multi-stage hybrid engine (`diff_engine.py`) for PDF documents and a vectorized matrix engine (`app.py`) for images and raster fallbacks.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 Browser Frontend                                 │
│  • Drag & Drop Uploads  • Synchronized Scrolling  • Dynamic Spotlights           │
│  • Slider Split View    • Flicker Mode Engine     • Differences Inspector Panel  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Multipart POST (/api/compare)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Flask Backend (app.py)                               │
│  • Validates file signatures (Magic bytes: %PDF-, PNG, JPEG, WebP, BMP, GIF)    │
│  • Dispatches PDF inputs to diff_engine.py or image inputs to compute_pixel_diff │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
┌───────────────────────────────────────┐ ┌────────────────────────────────────────┐
│      Hybrid PDF Engine (diff_engine)  │ │      Image Matrix Engine (app.py)      │
├───────────────────────────────────────┤ ├────────────────────────────────────────┤
│ Pass 0: MD5/SHA256 identity caching   │ │ 1. Pillow image decoding               │
│ Stage 1: Word-level text extraction   │ │ 2. Dimension alignment & white padding │
│   & Levenshtein alignment             │ │ 3. Vectorized RGB Euclidean diff:      │
│ Stage 2: Vector drawing comparison    │ │    |arr_a - arr_b| > sensitivity       │
│   with 2D spatial grid indexing       │ │ 4. SciPy ndimage morphological labeling│
│ Stage 3: Embedded raster image hash   │ │ 5. Bounding box & crop extraction      │
│ Stage 4: Residual raster pixel diff   │ │ 6. JSON serialization with base64 imgs │
│ Stage 5: Change clustering & crops    │ └────────────────────────────────────────┘
└───────────────────────────────────────┘
```

---

## ⚡ Dense Engineering & CAD Drawing Optimization

Technical schematics (e.g., Airbus aircraft wiring schematics, architectural CAD blueprints, and circuit layouts) often contain **over 9,000 vector drawing paths** per page.

A naive pairwise comparison algorithm has $O(N \times M)$ complexity. Comparing two pages with 9,200 paths each requires over **84,640,000 polygon distance checks**, causing requests to run for 13+ minutes or hit server gateway timeouts.

### How DiffView Solves This:
1. **Pass 0 Identity Caching**: Unmodified vector paths are hashed and matched in $O(N)$ time.
2. **2D Uniform Spatial Grid Index**:
   - The document surface is partitioned into uniform spatial grid cells ($128 \times 128\text{ px}$).
   - Drawing elements from Document B are indexed into spatial grid buckets based on their bounding coordinates in $O(M)$ time.
   - Drawing elements from Document A query only their localized neighboring buckets rather than iterating through all 9,200 elements.
3. **Performance Benchmark**:
   - **Airbus 6866VB Schematics (9,201 vector paths)**:
     - *Naive approach*: ~810 seconds (>13 minutes).
     - *DiffView 2D Spatial Grid*: **3.4 seconds** (99.6% reduction in latency).

---

## 📁 Directory Structure

```
DiffView - PDF & Image Comparator/
├── app.py                 # Flask server, API endpoints, file validation & raster diff
├── diff_engine.py         # Multi-stage PDF engine, 2D spatial grid & word text diff
├── requirements.txt       # Dependencies (Flask, PyMuPDF, Pillow, NumPy, SciPy)
├── README.md              # Comprehensive documentation & developer guide
├── .gitignore             # Standard git exclusions (venv, __pycache__, logs, etc.)
├── templates/
│   └── index.html         # Modern dark-mode interface with diff inspector
└── static/
    ├── style.css          # Design system tokens, animations, responsive layout
    └── app.js             # UI controller, synchronized scrolling & spotlight engine
```

---

## 🚀 Prerequisites & Installation

### Processing and Upload Limits
- Comparisons are performed entirely in memory; files are never written to disk.
- Uploads are limited to 128 MB per request (`MAX_CONTENT_LENGTH = 128 * 1024 * 1024`).
- Supported formats: PDF, PNG, JPG/JPEG, WebP, BMP, GIF, and TIFF.
- File signatures (magic bytes) are verified server-side.

### 1. Requirements
- **Python**: Version `3.10` or higher
- **pip**: Latest Python package manager

### 2. Clone the Repository
```bash
git clone https://github.com/AbhinavPandey12/DiffView-Image-PDF-Differentiator.git
cd "DiffView - PDF & Image Comparator"
```

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration & Environment Variables

You can configure DiffView using the following environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `5000` | Port number on which the Flask web server listens. |
| `FLASK_DEBUG` | `0` | Set to `1` to enable Flask auto-reloading and debug traceback. |
| `DIFFVIEW_MAX_DIMENSION` | `1400` | Maximum rendered width or height in pixels for comparison. |

Example:
```bash
# Run on port 8080 with high rendering resolution
PORT=8080 DIFFVIEW_MAX_DIMENSION=2000 python app.py
```

---

## 💻 How to Run & Use the Application

### 1. Start the Server
```bash
python app.py
```
You will see:
```text
============================================================
  DiffView — PDF & Image Comparator (Python Server)
  Application running at: http://127.0.0.1:5000
============================================================
```

### 2. Open in Browser
Navigate to `http://127.0.0.1:5000` in your web browser.

### 3. Compare Two Documents
1. **Upload Original (A)**: Drag and drop or browse for your base PDF or image in Box A.
2. **Upload Revised (B)**: Drag and drop or browse for your revised document in Box B.
3. **Click Compare**: Hit **Compare Documents**.
4. **Inspect Differences**:
   - Use the **Differences Breakdown & Written Analysis** panel below the viewer to review line-by-line text modifications and visual cards.
   - Click **Spotlight** on any card to locate and focus on that discrepancy in the viewer.
   - Switch between **Side-by-Side**, **Pixel Diff**, **Slider Split**, and **Flash Compare** modes.
   - Adjust the **Sensitivity** slider to tune tolerance.
   - Click **Export** to download the diff result as a PNG.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `1` | Switch to Side-by-Side view |
| `2` | Switch to Pixel Diff view |
| `3` | Switch to Slider Split view |
| `4` | Switch to Flash / Flicker view |
| `h` / `H` | Toggle visual bounding box highlights on/off |
| `[` | Navigate to Previous Difference |
| `]` | Navigate to Next Difference |
| `Space` | Press & hold to flash Document B over Document A (Flicker mode) |
| `+` / `=` | Zoom in ($+25\%$) |
| `-` | Zoom out ($-25\%$) |
| `0` | Reset zoom to $100\%$ |
| `←` / `→` | Previous / Next page (for multi-page PDFs) |

---

## 📡 API Reference

### 1. `POST /api/compare`
Uploads two files and computes visual differences, bounding boxes, text differences, and snippet previews.

**Parameters (`multipart/form-data`)**:
- `file_a`: (File, Required) Original document (PDF or image).
- `file_b`: (File, Required) Modified document (PDF or image).
- `page`: (Integer, Optional) Page number to render, 1-indexed (default: `1`).
- `sensitivity`: (Integer, Optional) Color difference threshold from `1` to `50` (default: `10`).

**Response (`application/json`)**:
```json
{
  "success": true,
  "changed_pct": 0.05,
  "same_pct": 99.95,
  "diff_pixels": 378,
  "total_pixels": 756000,
  "width": 1200,
  "height": 630,
  "current_page": 1,
  "total_pages": 1,
  "differences_count": 2,
  "is_completely_different": false,
  "bounding_boxes": [
    { "id": 1, "x": 118, "y": 450, "width": 180, "height": 28 },
    { "id": 2, "x": 640, "y": 320, "width": 95, "height": 40 }
  ],
  "diff_details": [
    {
      "id": 1,
      "x": 118,
      "y": 450,
      "width": 180,
      "height": 28,
      "change_type": "modified",
      "source": "text",
      "label": "Text Modified",
      "description": "Text changed from \"PB\" to \"13\".",
      "original_text": "PB",
      "revised_text": "13",
      "thumb_a_b64": "<base64_crop>",
      "thumb_b_b64": "<base64_crop>"
    }
  ],
  "text_diff": [
    {
      "type": "modified",
      "original": "PB",
      "revised": "13"
    }
  ],
  "img_a_b64": "<base64_png>",
  "img_b_b64": "<base64_png>",
  "img_a_highlighted_b64": "<base64_png>",
  "img_b_highlighted_b64": "<base64_png>",
  "diff_image_b64": "<base64_png>",
  "diff_3c_b64": "<base64_png>"
}
```

### 2. `GET /api/health`
Health check status endpoint.
```json
{
  "app": "DiffView PDF & Image Comparator",
  "status": "ok"
}
```

---

## 🔬 Algorithm Deep-Dive

### 1. Vectorized Pixel Difference
Let $I_A(x, y)$ and $I_B(x, y)$ represent the RGB color values of Document A and Document B at coordinate $(x, y)$:

$$\Delta(x, y) = \frac{1}{3} \sum_{c \in \{R,G,B\}} |I_A^{(c)}(x, y) - I_B^{(c)}(x, y)|$$

A pixel is marked as modified when:
$$\Delta(x, y) > \text{Sensitivity}$$

### 2. Morphological Clustering & Bounding Box Generation
To avoid overwhelming the user with thousands of isolated single pixels, DiffView applies morphological binary dilation using a $5 \times 5$ rectangular structuring element:
$$M_{\text{dilated}} = M \oplus B_{5 \times 5}$$

Connected components are then identified via SciPy:
```python
labeled_arr, num_features = ndimage.label(M_dilated)
objects = ndimage.find_objects(labeled_arr)
```
Each cluster meeting the minimum area threshold ($> 15\text{ px}^2$) is converted into a distinct bounding box `(x, y, width, height)`.

### 3. Word-Level Text Alignment
Text comparison in `diff_engine.py` extracts words with bounding coordinates `(x0, y0, x1, y1)`. For every word in Document A, the engine matches candidates in Document B using:
$$\text{Cost} = \alpha \cdot \text{NormalizedSpatialDistance} + (1 - \alpha) \cdot \text{LevenshteinDistance}$$
This reliably separates true textual modifications from layout reflows and paragraph shifts.

### 4. Change Classification
For every bounding box, the mean luminance of the crop in Document A ($\mu_A$) and Document B ($\mu_B$) is analyzed:
- $\mu_A < 235$ and $\mu_B \ge 215 \implies$ **Removed from A** (content disappeared against white background).
- $\mu_A \ge 215$ and $\mu_B < 235 \implies$ **Added in B** (new visual content drawn on blank area).
- Otherwise $\implies$ **Content Modified** (color, shape, or typography altered).

---

## 🧪 Testing & Quality Assurance

A comprehensive test suite is included to guarantee correctness across rendering, vector path indexing, and highlight generation:

```bash
# Run highlight bounding box and drawing regression tests
python scratch/test_highlights.py

# Run CAD drawing engine benchmark
python scratch/test_cad_diff.py

# Verify end-to-end HTTP API upload and comparison
python scratch/test_user_api.py
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute for commercial or non-commercial purposes.
