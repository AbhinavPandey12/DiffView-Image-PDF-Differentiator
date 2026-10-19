# 📄 DiffView — PDF & Image Visual & Textual Comparator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/PyMuPDF-High_Performance-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="PyMuPDF"/>
  <img src="https://img.shields.io/badge/NumPy-Vectorized_Diff-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/SciPy-Cluster_BBoxes-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/Pillow-Image_Processing-945DD6?style=for-the-badge" alt="Pillow"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

> **DiffView** is an open-source visual document and image comparison web application. It combines high-speed vector rasterization, morphological computer vision clustering, and line-level text diffing to spot subtle modifications, design adjustments, typography edits, and visual alterations with pixel-level precision.
> Designed with the modern dark-mode aesthetic of [ImgZero Compare PDFs](https://www.imgzero.in/compare-pdfs).

---

## 📑 Table of Contents

- [Overview & Key Features](#-overview--key-features)
- [Exclusive Difference Inspector](#-exclusive-difference-inspector)
- [Comparison Modes Breakdown](#-comparison-modes-breakdown)
- [How It Works (Technical Architecture)](#-how-it-works-technical-architecture)
- [Directory Structure](#-directory-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [How to Run & Use the Application](#-how-to-run--use-the-application)
- [Keyboard Shortcuts](#-keyboard-shortcuts)
- [API Reference](#-api-reference)
- [Algorithm Deep-Dive](#-algorithm-deep-dive)
- [Contributing & License](#-contributing--license)

---

## ✨ Overview & Key Features

DiffView bridges the gap between raw document diffing and visual QA by analyzing pixel differences across rendered PDF pages and bitmap images:

- 🎯 **Pixel-Level Precision**: Highlights every altered pixel with configurable sensitivity and change percentage statistics.
- 🔍 **Exclusive Differences Inspector**: A dedicated analytical component that breaks down and writes out all textual revisions and visual changes with side-by-side snippet crops and animated spotlights.
- 📑 **Full Multi-Page PDF Support**: Seamlessly render and navigate through multi-page PDFs using high-speed PyMuPDF rasterization.
- 🖼️ **Cross-Format Flexibility**: Compare `PDF`, `PNG`, `JPG`, `JPEG`, `WebP`, `BMP`, and `GIF` files interchangeably (e.g. PDF vs. PDF or Image vs. Image).
- 🎛️ **4 Intuitive Comparison Modes**:
  1. **Side-by-Side**: Synchronized scrolling panes showing Document A (with red highlights) and Document B (with green highlights).
  2. **Pixel Diff (Heatmap & 3-Color)**: High-contrast red overlay or 3-color breakdown (*Red = Removed, Green = Added, Amber = Modified*).
  3. **Slider Split Screen**: Interactive before-and-after drag-wipe slider overlay with adjustable opacity.
  4. **Flash / Flicker Mode**: Rapid A/B alternating flash on button hold or `Spacebar` to spot micro layout jitters.
- 🎚️ **Dynamic Sensitivity Slider**: Tune the color distance tolerance ($1 - 50$) on the fly to eliminate compression noise or capture microscopic sub-pixel changes.
- 🔍 **Interactive Zoom & Pan**: Zoom smoothly from $25\%$ to $300\%$ with keyboard shortcuts and synchronized panning.
- 💾 **Export Results**: Download high-resolution PNG diff visualizations directly to your computer.
- 🌓 **Dark / Light Mode**: Sleek dark interface with toggleable light mode persisted in `localStorage`.

---

## 🔎 Exclusive Difference Inspector

The application features a dedicated, exclusive analytical panel (**#diff-inspector**) positioned below the viewer for comprehensive auditing:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Differences Breakdown & Written Analysis                              │
│  [Total: 4]   [Removed: 2]   [Added: 1]   [Modified: 1]                │
├────────────────────────────────────────────────────────────────────────┤
│  Written Text Comparison (Line-by-Line)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ [MODIFIED] Text Revision #1                                      │  │
│  │ - Original (A): Clause 1: Payment is due within 30 days.         │  │
│  │ + Revised  (B): Clause 1: Payment is due within 15 days.         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────┤
│  Visual & Graphical Differences Log                                    │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────┐  │
│  │ Diff #1  [REMOVED]              │  │ Diff #2  [ADDED]            │  │
│  │ X: 120, Y: 450 (80×30px)        │  │ X: 350, Y: 180 (110×45px)   │  │
│  │ Element removed in Document B   │  │ New content added in Doc B  │  │
│  │ ┌─────────────┐ ┌─────────────┐ │  │ ┌───────────┐ ┌───────────┐ │  │
│  │ │ Doc A Crop  │ │ Doc B Crop  │ │  │ │Doc A Crop │ │Doc B Crop │ │  │
│  │ └─────────────┘ └─────────────┘ │  │ └───────────┘ └───────────┘ │  │
│  │ [Spotlight Button]              │  │ [Spotlight Button]          │  │
│  └─────────────────────────────────┘  └─────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Highlights of the Inspector:
1. **Line-by-Line Text Comparison (for PDFs)**:
   - Uses PyMuPDF text stream extraction and Python's `difflib.SequenceMatcher` to compare line-level textual revisions.
   - Distinctly renders red original lines (`-`) and green revised lines (`+`).
2. **Visual Differences Cards Grid (for both Images and PDFs)**:
   - Every visually detected discrepancy cluster is assigned a card with bounding coordinates `(X, Y, W×H px)`.
   - Generates high-resolution cropped snippet thumbnails comparing Document A (red border) vs. Document B (green border).
   - Writes an explicit description explaining whether the element was removed, added, or modified.
3. **Animated Spotlight Navigation**:
   - Clicking any card or difference in the navigator triggers an animated pulsing spotlight box (`.diff-spotlight`) directly over the difference on the document view, and smoothly scrolls to center the discrepancy.

---

## 🎛️ Comparison Modes Breakdown

| Mode | Visual Mechanism | Best Used For |
| :--- | :--- | :--- |
| **Side-by-Side** | Dual synchronized panes with red (Doc A) and green (Doc B) bounding boxes & spotlight pulse | Layout comparison, typography checking, verifying margin shifts |
| **Pixel Diff (Heatmap)** | High-contrast opaque white background with glowing red diff mask and bounding boxes | Identifying every single pixel discrepancy across complex documents |
| **Pixel Diff (3-Color)** | 3-Color classification: Red (Removed from A), Green (Added in B), Amber (Modified) | Auditing contract clauses, legal documents, engineering drawings |
| **Slider Split Screen** | Interactive before-and-after drag wipe with adjustable opacity slider | Validating alignment, stamp placement, watermark changes |
| **Flash / Flicker Mode** | Rapid A/B alternating flash on hold button or `Spacebar` | Spotting minute single-pixel shifts, subtle font weight discrepancies |
| **Differences Inspector** | Dedicated breakdown panel with written text diffs, visual cards & snippet crops | Item-by-item change verification and client audit reporting |

---

## 🛠️ How It Works (Technical Architecture)

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
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Ingestion: Sanitizes byte streams, detects PDF magic header (%PDF-) vs Images │
│ 2. Rendering: PyMuPDF rasterizes PDF pages with anti-aliasing scaling matrix     │
│ 3. Alignment: Pads both images to max dimensions (W, H) with white background    │
│ 4. Matrix Diff: NumPy vectorized RGB channel distance: |arr_a - arr_b| > sens    │
│ 5. Morphological Grouping: SciPy ndimage.label & find_objects group diff clusters│
│ 6. Text Diff: PyMuPDF get_text() + difflib.SequenceMatcher line diffing          │
│ 7. Snippet Crops: Pillow crops bounding boxes into base64 thumbnail previews     │
│ 8. JSON Response: Encodes images to Base64 + returns full diff_details & stats   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
PDF & Image Comparator/
├── app.py                 # Flask server, PyMuPDF engine, NumPy diff & text diff
├── requirements.txt       # Dependencies (Flask, PyMuPDF, Pillow, NumPy, SciPy)
├── README.md              # Comprehensive project documentation & usage guide
├── .gitignore             # Standard git exclusions (venv, __pycache__, etc.)
├── templates/
│   └── index.html         # Modern dark-mode interface with diff inspector
└── static/
    ├── style.css          # Design tokens, spotlight pulse animation, cards grid
    └── app.js             # UI controller, synchronized scrolling, spotlight logic
```

---

## 🚀 Prerequisites & Installation

### 1. Requirements
- **Python**: Version `3.10` or higher
- **pip**: Latest Python package manager

### 2. Clone the Repository
```bash
git clone https://github.com/AbhinavPandey12/ShopEase.git
cd "PDF & Image Comparator"
```

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
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
3. **Click Compare**: Hit **Compare Files**.
4. **Inspect Differences**:
   - Use the **Differences Inspector** below the viewer to read line-by-line text differences and inspect side-by-side cropped snippets.
   - Click **Spotlight** on any difference card to locate and focus on that discrepancy in the viewer.
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
Uploads two files and computes the visual diff, bounding boxes, text diff, and snippet previews.

**Parameters (`multipart/form-data`)**:
- `file_a`: (File, Required) Original document (PDF or image).
- `file_b`: (File, Required) Modified document (PDF or image).
- `page`: (Integer, Optional) Page number to render (default: `1`).
- `sensitivity`: (Integer, Optional) Color difference threshold from `1` to `50` (default: `10`).

**Response (`application/json`)**:
```json
{
  "success": true,
  "changed_pct": 3.45,
  "same_pct": 96.55,
  "diff_pixels": 7820,
  "total_pixels": 226800,
  "width": 600,
  "height": 378,
  "current_page": 1,
  "total_pages": 1,
  "differences_count": 2,
  "bounding_boxes": [
    { "id": 1, "x": 120, "y": 45, "width": 80, "height": 30 },
    { "id": 2, "x": 280, "y": 140, "width": 65, "height": 25 }
  ],
  "diff_details": [
    {
      "id": 1,
      "x": 120,
      "y": 45,
      "width": 80,
      "height": 30,
      "change_type": "removed",
      "label": "Removed from Document A",
      "description": "Element previously located at (X: 120, Y: 45) was removed in Document B.",
      "thumb_a_b64": "<base64_crop>",
      "thumb_b_b64": "<base64_crop>"
    }
  ],
  "text_diff": [
    {
      "type": "modified",
      "original": "Clause 1: Payment is due within 30 days.",
      "revised": "Clause 1: Payment is due within 15 days."
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
Let $I_A(x, y)$ and $I_B(x, y)$ represent the RGB values of Document A and Document B at coordinate $(x, y)$:

$$\Delta(x, y) = \frac{1}{3} \sum_{c \in \{R,G,B\}} |I_A^{(c)}(x, y) - I_B^{(c)}(x, y)|$$

A pixel is marked as changed when:
$$\Delta(x, y) > \text{Sensitivity}$$

### 2. Morphological Clustering & Bounding Box Generation
To avoid overwhelming the user with thousands of disconnected single pixels, DiffView applies morphological binary dilation using a $5 \times 5$ structuring element:
$$M_{\text{dilated}} = M \oplus B_{5 \times 5}$$

Connected components are then identified via SciPy:
```python
labeled_arr, num_features = ndimage.label(M_dilated)
objects = ndimage.find_objects(labeled_arr)
```
Each cluster meeting the minimum area threshold ($> 15\text{ px}^2$) is converted into a distinct bounding box `(x, y, width, height)`.

### 3. Change Classification
For every bounding box, the mean luminance of the crop in Document A ($\mu_A$) and Document B ($\mu_B$) is analyzed:
- $\mu_A < 235$ and $\mu_B \ge 215 \implies$ **Removed from A** (content vanished against white background).
- $\mu_A \ge 215$ and $\mu_B < 235 \implies$ **Added in B** (new visual content drawn on blank area).
- Otherwise $\implies$ **Content Modified** (color, shape, or typography altered).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute for commercial or non-commercial purposes.
