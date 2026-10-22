/**
 * DiffView — Client-Side Application Controller
 * Handles document upload, Flask API requests, side-by-side synchronized viewing,
 * highlighted diff overlays, 3-color add/remove diffing, interactive split slider,
 * flash/flicker mode, difference navigator, zoom, and export.
 */

(function () {
  'use strict';

  // ─────────────────────────────────────────────────────────────
  // STATE MANAGEMENT
  // ─────────────────────────────────────────────────────────────
  const state = {
    fileA: null,
    fileB: null,
    currentPage: 1,
    totalPages: 1,
    sensitivity: 10,
    zoomLevel: 100, // percentage
    activeView: 'side-by-side',
    isComparing: false,
    diffData: null,
    isDraggingSlider: false,

    // Difference inspection
    showHighlights: true,
    currentDiffIndex: 0,
    diffSubmode: 'heatmap', // 'heatmap' or '3color'

    // Flicker state
    isFlickeringB: false,
  };

  // ─────────────────────────────────────────────────────────────
  // DOM SELECTORS
  // ─────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const els = {
    // Header & Theme
    themeToggle: $('theme-toggle'),

    // Upload Zones
    cardA: $('card-a'),
    cardB: $('card-b'),
    dropA: $('drop-a'),
    dropB: $('drop-b'),
    inpA: $('inp-a'),
    inpB: $('inp-b'),
    fileChosenA: $('file-chosen-a'),
    fileChosenB: $('file-chosen-b'),
    nameA: $('name-a'),
    nameB: $('name-b'),
    rmA: $('rm-a'),
    rmB: $('rm-b'),

    // Action Controls
    btnCompare: $('btn-compare'),
    compareHint: $('compare-hint'),
    btnSwap: $('btn-swap'),

    // Loading State
    loadingWrap: $('loading-wrap'),
    progressFill: $('progress-fill'),
    loadingLabel: $('loading-label'),

    // Workspace
    workspace: $('workspace'),
    vtabSBS: $('vtab-sbs'),
    vtabDiff: $('vtab-diff'),
    vtabOV: $('vtab-ov'),
    vtabFlicker: $('vtab-flicker'),

    viewSBS: $('view-sbs'),
    viewDiff: $('view-diff'),
    viewOV: $('view-ov'),
    viewFlicker: $('view-flicker'),

    // Toolbar Controls
    btnToggleHighlights: $('btn-toggle-highlights'),
    diffNavigator: $('diff-navigator'),
    diffPrev: $('diff-prev'),
    diffNext: $('diff-next'),
    diffNavLabel: $('diff-nav-label'),

    pageCtrl: $('page-ctrl'),
    pgPrev: $('pg-prev'),
    pgNext: $('pg-next'),
    pgInput: $('pg-input'),
    pgTotal: $('pg-total'),

    sensCtrl: $('sens-ctrl'),
    sensSlider: $('sens-slider'),
    sensVal: $('sens-val'),

    zoomOut: $('zoom-out'),
    zoomIn: $('zoom-in'),
    zoomReset: $('zoom-reset'),
    zoomLabel: $('zoom-label'),
    btnExport: $('btn-export'),

    // Stats Bar
    sChanged: $('s-changed'),
    sSame: $('s-same'),
    sCount: $('s-count'),
    sDims: $('s-dims'),
    sPx: $('s-px'),

    // View Contents
    scrollA: $('scroll-a'),
    scrollB: $('scroll-b'),
    imgA: $('img-a'),
    imgB: $('img-b'),
    lblA: $('lbl-a'),
    lblB: $('lbl-b'),
    imgDiff: $('img-diff'),

    // Diff Submodes
    smHeatmap: $('sm-heatmap'),
    sm3color: $('sm-3color'),
    legAdded: $('leg-added'),
    legMod: $('leg-mod'),

    // Slider Split Overlay
    ovContainer: $('ov-container'),
    ovImgA: $('ov-img-a'),
    ovImgB: $('ov-img-b'),
    ovClip: $('ov-clip'),
    ovLine: $('ov-line'),
    ovHandle: $('ov-handle'),
    ovOpacity: $('ov-opacity'),
    ovOpval: $('ov-opval'),

    // Flash / Flicker Mode
    imgFlicker: $('img-flicker'),
    flickerBadge: $('flicker-state-badge'),
    btnFlickerHold: $('btn-flicker-hold'),

    // Exclusive Differences Inspector & Spotlights
    diffInspector: $('diff-inspector'),
    diTotalCount: $('di-total-count'),
    diDelCount: $('di-del-count'),
    diAddCount: $('di-add-count'),
    diModCount: $('di-mod-count'),
    diTextSection: $('di-text-section'),
    diTextList: $('di-text-list'),
    diVisualSection: $('di-visual-section'),
    diCardsGrid: $('di-cards-grid'),
    spotlightA: $('spotlight-a'),
    spotlightB: $('spotlight-b'),
    wrapA: $('wrap-a'),
    wrapB: $('wrap-b'),
  };

  // ─────────────────────────────────────────────────────────────
  // INITIALIZATION
  // ─────────────────────────────────────────────────────────────
  function init() {
    setupTheme();
    setupDropzone(els.dropA, els.inpA, (f) => setFile('A', f));
    setupDropzone(els.dropB, els.inpB, (f) => setFile('B', f));

    els.rmA.addEventListener('click', (e) => {
      e.stopPropagation();
      clearFile('A');
    });

    els.rmB.addEventListener('click', (e) => {
      e.stopPropagation();
      clearFile('B');
    });

    els.btnSwap.addEventListener('click', swapFiles);
    els.btnCompare.addEventListener('click', () => runComparison(true));

    // View Switching
    els.vtabSBS.addEventListener('click', () => switchView('side-by-side'));
    els.vtabDiff.addEventListener('click', () => switchView('diff'));
    els.vtabOV.addEventListener('click', () => switchView('overlay'));
    els.vtabFlicker.addEventListener('click', () => switchView('flicker'));

    // Highlight Toggle on Side-by-Side
    els.btnToggleHighlights.addEventListener('click', toggleHighlights);

    // Difference Navigation
    els.diffPrev.addEventListener('click', () => navigateDiff(-1));
    els.diffNext.addEventListener('click', () => navigateDiff(1));

    // Pixel Diff Submodes
    els.smHeatmap.addEventListener('click', () => setDiffSubmode('heatmap'));
    els.sm3color.addEventListener('click', () => setDiffSubmode('3color'));

    // Pagination
    els.pgPrev.addEventListener('click', () => goToPage(state.currentPage - 1));
    els.pgNext.addEventListener('click', () => goToPage(state.currentPage + 1));
    els.pgInput.addEventListener('change', (e) => {
      const p = parseInt(e.target.value, 10);
      if (!isNaN(p)) goToPage(p);
    });

    // Sensitivity Slider
    els.sensSlider.addEventListener('input', (e) => {
      state.sensitivity = parseInt(e.target.value, 10);
      els.sensVal.textContent = state.sensitivity;
    });
    els.sensSlider.addEventListener('change', () => {
      if (state.fileA && state.fileB) {
        runComparison(false);
      }
    });

    // Zoom Controls
    els.zoomIn.addEventListener('click', () => adjustZoom(25));
    els.zoomOut.addEventListener('click', () => adjustZoom(-25));
    els.zoomReset.addEventListener('click', () => setZoom(100));

    // Synchronized Scrolling for Side-by-Side Panes
    setupSynchronizedScroll();

    // Split Slider Dragging
    setupSplitSlider();

    // Overlay Opacity
    els.ovOpacity.addEventListener('input', (e) => {
      const val = e.target.value;
      els.ovOpval.textContent = `${val}%`;
      els.ovImgB.style.opacity = (val / 100).toString();
    });

    // Flash / Flicker Mode
    setupFlickerMode();

    // Export diff image
    els.btnExport.addEventListener('click', exportDiffResult);

    // Global Keyboard Shortcuts
    document.addEventListener('keydown', handleKeyShortcuts);
    document.addEventListener('keyup', handleKeyUp);
  }

  // ─────────────────────────────────────────────────────────────
  // THEME
  // ─────────────────────────────────────────────────────────────
  function setupTheme() {
    const saved = localStorage.getItem('diffview_theme') || 'dark';
    document.body.className = saved;

    els.themeToggle.addEventListener('click', () => {
      const isDark = document.body.classList.contains('dark');
      const newTheme = isDark ? 'light' : 'dark';
      document.body.className = newTheme;
      localStorage.setItem('diffview_theme', newTheme);
    });
  }

  // ─────────────────────────────────────────────────────────────
  // FILE HANDLING & DRAG-AND-DROP
  // ─────────────────────────────────────────────────────────────
  function setupDropzone(dropzone, input, onFile) {
    dropzone.addEventListener('click', () => input.click());

    input.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        onFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFile(e.dataTransfer.files[0]);
      }
    });
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function setFile(which, file) {
    if (which === 'A') {
      state.fileA = file;
      els.nameA.textContent = `${file.name} (${formatFileSize(file.size)})`;
      els.fileChosenA.hidden = false;
      els.dropA.classList.add('has-file');
      els.lblA.textContent = file.name;
    } else {
      state.fileB = file;
      els.nameB.textContent = `${file.name} (${formatFileSize(file.size)})`;
      els.fileChosenB.hidden = false;
      els.dropB.classList.add('has-file');
      els.lblB.textContent = file.name;
    }

    updateCompareButtonState();
  }

  function clearFile(which) {
    if (which === 'A') {
      state.fileA = null;
      els.inpA.value = '';
      els.nameA.textContent = '';
      els.fileChosenA.hidden = true;
      els.dropA.classList.remove('has-file');
    } else {
      state.fileB = null;
      els.inpB.value = '';
      els.nameB.textContent = '';
      els.fileChosenB.hidden = true;
      els.dropB.classList.remove('has-file');
    }

    updateCompareButtonState();
  }

  function swapFiles() {
    if (!state.fileA && !state.fileB) return;

    const temp = state.fileA;
    state.fileA = state.fileB;
    state.fileB = temp;

    if (state.fileA) {
      els.nameA.textContent = `${state.fileA.name} (${formatFileSize(state.fileA.size)})`;
      els.fileChosenA.hidden = false;
      els.dropA.classList.add('has-file');
      els.lblA.textContent = state.fileA.name;
    } else {
      clearFile('A');
    }

    if (state.fileB) {
      els.nameB.textContent = `${state.fileB.name} (${formatFileSize(state.fileB.size)})`;
      els.fileChosenB.hidden = false;
      els.dropB.classList.add('has-file');
      els.lblB.textContent = state.fileB.name;
    } else {
      clearFile('B');
    }

    updateCompareButtonState();

    if (state.fileA && state.fileB && !els.workspace.hidden) {
      runComparison(false);
    }
  }

  function updateCompareButtonState() {
    const ready = Boolean(state.fileA && state.fileB);
    els.btnCompare.disabled = !ready;
    if (ready) {
      els.compareHint.textContent = 'Both documents ready. Click Compare Files.';
      els.compareHint.style.color = 'var(--success)';
    } else {
      els.compareHint.textContent = 'Upload both files to start comparison';
      els.compareHint.style.color = 'var(--subtle)';
    }
  }

  // ─────────────────────────────────────────────────────────────
  // COMPARISON EXECUTION
  // ─────────────────────────────────────────────────────────────
  async function runComparison(resetPage = true) {
    if (!state.fileA || !state.fileB) return;

    if (resetPage) {
      state.currentPage = 1;
      state.currentDiffIndex = 0;
    }

    setLoading(true, 'Analyzing documents and computing visual diff...');

    const formData = new FormData();
    formData.append('file_a', state.fileA);
    formData.append('file_b', state.fileB);
    formData.append('page', state.currentPage);
    formData.append('sensitivity', state.sensitivity);

    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errMsg = `Server error (${response.status})`;
        try {
          const err = await response.json();
          if (err && err.error) errMsg = err.error;
        } catch {
          const text = await response.text();
          if (text) {
            const clean = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
            if (clean) errMsg = clean.slice(0, 160);
          }
        }
        throw new Error(errMsg);
      }

      let data;
      try {
        data = await response.json();
      } catch (jsonErr) {
        throw new Error('Server returned an unexpected response format. Please try again.');
      }

      state.diffData = data;
      renderComparisonResults(data);
    } catch (err) {
      alert(`Error comparing documents: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading, message = 'Processing...') {
    state.isComparing = loading;
    els.loadingWrap.hidden = !loading;
    els.loadingLabel.textContent = message;
    els.btnCompare.disabled = loading || !state.fileA || !state.fileB;
  }

  // ─────────────────────────────────────────────────────────────
  // RENDER RESULTS
  // ─────────────────────────────────────────────────────────────
  function renderComparisonResults(data) {
    els.workspace.hidden = false;

    // Update Stats Bar
    if (data.is_completely_different) {
      els.sChanged.textContent = `${data.changed_pct}%`;
      els.sSame.textContent = `${data.same_pct}%`;
      els.sCount.textContent = 'Completely Different';
      els.sDims.textContent = `${data.width} × ${data.height} px`;
      els.sPx.textContent = Number(data.diff_pixels).toLocaleString();
      state.showHighlights = false;
      if (els.btnToggleHighlights) {
        els.btnToggleHighlights.classList.remove('active');
        els.btnToggleHighlights.classList.add('off');
        els.btnToggleHighlights.title = 'Highlights disabled: sources are completely different';
        els.btnToggleHighlights.style.opacity = '0.5';
        els.btnToggleHighlights.style.pointerEvents = 'none';
        els.btnToggleHighlights.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          Highlights: N/A
        `;
      }
    } else {
      els.sChanged.textContent = `${data.changed_pct}%`;
      els.sSame.textContent = `${data.same_pct}%`;
      els.sCount.textContent = (data.differences_count || 0).toString();
      els.sDims.textContent = `${data.width} × ${data.height} px`;
      els.sPx.textContent = Number(data.diff_pixels).toLocaleString();
      if (els.btnToggleHighlights) {
        els.btnToggleHighlights.style.opacity = '';
        els.btnToggleHighlights.style.pointerEvents = '';
        if (state.showHighlights) {
          els.btnToggleHighlights.classList.add('active');
          els.btnToggleHighlights.classList.remove('off');
          els.btnToggleHighlights.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/></svg>
            Highlights: ON
          `;
        }
      }
    }

    // Populate Side-by-Side
    updateSideBySideImages();

    // Populate Pixel Diff (Heatmap default)
    setDiffSubmode(state.diffSubmode);

    // Populate Slider Split
    const srcA = `data:image/png;base64,${data.img_a_b64}`;
    const srcB = `data:image/png;base64,${data.img_b_b64}`;
    els.ovImgA.src = srcA;
    els.ovImgB.src = srcB;

    // Populate Flash / Flicker Mode
    els.imgFlicker.src = srcA;
    state.isFlickeringB = false;
    els.flickerBadge.textContent = 'Showing: Document A (Original)';

    // Multi-page Controls
    state.totalPages = data.total_pages || 1;
    if (state.totalPages > 1) {
      els.pageCtrl.hidden = false;
      els.pgInput.value = state.currentPage;
      els.pgInput.max = state.totalPages;
      els.pgTotal.textContent = state.totalPages;
      els.pgPrev.disabled = state.currentPage <= 1;
      els.pgNext.disabled = state.currentPage >= state.totalPages;
    } else {
      els.pageCtrl.hidden = true;
    }

    // Difference Navigator
    const count = (data.bounding_boxes || []).length;
    if (data.is_completely_different || count === 0) {
      els.diffNavLabel.textContent = data.is_completely_different ? 'Completely Different' : '0 Differences';
      els.diffNavigator.style.display = 'none';
    } else {
      state.currentDiffIndex = 0;
      els.diffNavLabel.textContent = `Diff 1 of ${count}`;
      els.diffNavigator.style.display = 'inline-flex';
    }

    // Reset overlay split position to 50%
    setOverlaySplitPosition(50);

    // Hide spotlights initially until a diff is focused
    if (els.spotlightA) els.spotlightA.hidden = true;
    if (els.spotlightB) els.spotlightB.hidden = true;

    // Render Exclusive Differences Inspector & Written Analysis
    renderDiffInspector(data);

    // Smooth scroll to workspace
    els.workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ─────────────────────────────────────────────────────────────
  // HIGHLIGHTS & SIDE-BY-SIDE VIEW
  // ─────────────────────────────────────────────────────────────
  function toggleHighlights() {
    if (state.diffData && state.diffData.is_completely_different) {
      return;
    }
    state.showHighlights = !state.showHighlights;

    if (state.showHighlights) {
      els.btnToggleHighlights.classList.remove('off');
      els.btnToggleHighlights.classList.add('active');
      els.btnToggleHighlights.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/></svg>
        Highlights: ON
      `;
    } else {
      els.btnToggleHighlights.classList.remove('active');
      els.btnToggleHighlights.classList.add('off');
      els.btnToggleHighlights.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
        Highlights: OFF
      `;
    }

    updateSideBySideImages();
  }

  function updateSideBySideImages() {
    if (!state.diffData) return;

    if (state.diffData.is_completely_different) {
      els.imgA.src = `data:image/png;base64,${state.diffData.img_a_b64}`;
      els.imgB.src = `data:image/png;base64,${state.diffData.img_b_b64}`;
      return;
    }

    if (state.showHighlights) {
      els.imgA.src = `data:image/png;base64,${state.diffData.img_a_highlighted_b64 || state.diffData.img_a_b64}`;
      els.imgB.src = `data:image/png;base64,${state.diffData.img_b_highlighted_b64 || state.diffData.img_b_b64}`;
    } else {
      els.imgA.src = `data:image/png;base64,${state.diffData.img_a_b64}`;
      els.imgB.src = `data:image/png;base64,${state.diffData.img_b_b64}`;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // DIFFERENCE NAVIGATOR & SPOTLIGHT FOCUS
  // ─────────────────────────────────────────────────────────────
  function navigateDiff(delta) {
    if (!state.diffData || !state.diffData.bounding_boxes || state.diffData.bounding_boxes.length === 0) return;

    const bboxes = state.diffData.bounding_boxes;
    const total = bboxes.length;
    let nextIdx = state.currentDiffIndex + delta;

    if (nextIdx < 0) nextIdx = total - 1;
    if (nextIdx >= total) nextIdx = 0;

    focusDifference(bboxes[nextIdx].id);
  }

  function focusDifference(diffId) {
    if (!state.diffData) return;
    const bboxes = state.diffData.bounding_boxes || [];
    const idx = bboxes.findIndex((b) => b.id === diffId);
    if (idx === -1) return;

    state.currentDiffIndex = idx;
    els.diffNavLabel.textContent = `Diff ${idx + 1} of ${bboxes.length}`;

    // Switch to side-by-side if not currently active
    if (state.activeView !== 'side-by-side') {
      switchView('side-by-side');
    }

    // Highlight active card in diff inspector
    const cards = els.diCardsGrid.querySelectorAll('.di-card');
    cards.forEach((c) => {
      if (parseInt(c.dataset.diffId, 10) === diffId) {
        c.classList.add('active');
        c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        c.classList.remove('active');
      }
    });

    const bbox = bboxes[idx];
    updateSpotlight(bbox);

    // Center scroll on the bounding box in both panes
    const scale = state.zoomLevel / 100;
    const baseW = els.imgA.clientWidth || state.diffData.width || 1;
    const baseH = els.imgA.clientHeight || state.diffData.height || 1;
    const ratioX = baseW / (state.diffData.width || 1);
    const ratioY = baseH / (state.diffData.height || 1);

    const paneHeight = els.scrollA.clientHeight || 500;
    const paneWidth = els.scrollA.clientWidth || 500;
    const targetY = Math.max(0, bbox.y * ratioY * scale - paneHeight / 3);
    const targetX = Math.max(0, bbox.x * ratioX * scale - paneWidth / 3);

    els.scrollA.scrollTo({ top: targetY, left: targetX, behavior: 'smooth' });
    els.scrollB.scrollTo({ top: targetY, left: targetX, behavior: 'smooth' });
  }

  function updateSpotlight(bbox) {
    if (!bbox || !els.spotlightA || !els.spotlightB || !state.diffData) return;

    const natW = state.diffData.width || 1;
    const natH = state.diffData.height || 1;
    const baseW = els.imgA.clientWidth || natW;
    const baseH = els.imgA.clientHeight || natH;
    const ratioX = baseW / natW;
    const ratioY = baseH / natH;

    const x = bbox.x * ratioX;
    const y = bbox.y * ratioY;
    const w = bbox.width * ratioX;
    const h = bbox.height * ratioY;
    const pad = 4;

    const apply = (sp) => {
      sp.hidden = false;
      sp.style.left = `${Math.max(0, x - pad)}px`;
      sp.style.top = `${Math.max(0, y - pad)}px`;
      sp.style.width = `${w + pad * 2}px`;
      sp.style.height = `${h + pad * 2}px`;
    };

    apply(els.spotlightA);
    apply(els.spotlightB);
  }

  // ─────────────────────────────────────────────────────────────
  // EXCLUSIVE ELEMENT: DIFFERENCE INSPECTOR & WRITTEN ANALYSIS
  // ─────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderDiffInspector(data) {
    const cdNotice = $('di-completely-different-notice');
    const badges = document.querySelector('.di-badges');

    if (data.is_completely_different) {
      if (cdNotice) cdNotice.style.display = 'flex';
      if (badges) badges.style.display = 'none';

      if (els.diTextSection) {
        els.diTextSection.hidden = true;
        els.diTextList.innerHTML = '';
      }
      if (els.diVisualSection) {
        els.diVisualSection.hidden = true;
        els.diVisualSection.style.display = 'none';
      }
      return;
    }

    if (cdNotice) cdNotice.style.display = 'none';
    if (badges) badges.style.display = 'flex';
    if (els.diVisualSection) {
      els.diVisualSection.hidden = false;
      els.diVisualSection.style.display = 'block';
    }

    const diffs = data.diff_details || [];
    const textDiffs = data.text_diff || [];
    const count = data.differences_count || 0;

    // Header Badges
    els.diTotalCount.textContent = count.toString();

    let delCount = 0;
    let addCount = 0;
    let modCount = 0;

    diffs.forEach((d) => {
      if (d.change_type === 'removed') delCount++;
      else if (d.change_type === 'added') addCount++;
      else modCount++;
    });

    els.diDelCount.textContent = delCount.toString();
    els.diAddCount.textContent = addCount.toString();
    els.diModCount.textContent = modCount.toString();

    // 1. Written Text Differences (Line-by-Line comparison)
    if (textDiffs && textDiffs.length > 0) {
      els.diTextSection.hidden = false;
      els.diTextList.innerHTML = '';
      textDiffs.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'di-text-card';
        const typeClass =
          item.type === 'modified'
            ? 'type-modified'
            : item.type === 'deleted'
            ? 'type-deleted'
            : 'type-added';
        const typeLabel = (item.type || 'MODIFIED').toUpperCase();

        let compHtml = '';
        if (item.original) {
          compHtml += `<div class="text-line-orig"><span style="user-select:none;opacity:0.65;font-weight:700;margin-right:6px;">Original (A):</span>${escapeHtml(item.original)}</div>`;
        }
        if (item.revised) {
          compHtml += `<div class="text-line-rev"><span style="user-select:none;opacity:0.65;font-weight:700;margin-right:6px;">Revised (B):</span>${escapeHtml(item.revised)}</div>`;
        }

        card.innerHTML = `
          <div class="di-text-header">
            <span class="di-type-badge ${typeClass}">${typeLabel}</span>
            <span style="font-size:11.5px;color:var(--subtle);font-weight:600;">Text Revision #${idx + 1}</span>
          </div>
          <div class="di-text-comparison">
            ${compHtml}
          </div>
        `;
        els.diTextList.appendChild(card);
      });
    } else {
      els.diTextSection.hidden = true;
      els.diTextList.innerHTML = '';
    }

    // 2. Visual Differences Cards Grid
    els.diCardsGrid.innerHTML = '';
    if (diffs.length > 0) {
      diffs.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'di-card';
        card.dataset.diffId = item.id;

        const badgeClass =
          item.change_type === 'removed'
            ? 'type-deleted'
            : item.change_type === 'added'
            ? 'type-added'
            : 'type-modified';

        card.innerHTML = `
          <div class="di-card-top">
            <div style="display:flex;align-items:center;gap:6px;">
              <span class="di-card-id">Diff #${item.id}</span>
              <span class="di-type-badge ${badgeClass}">${escapeHtml(item.change_type.toUpperCase())}</span>
            </div>
            <span class="di-card-loc">X: ${item.x}, Y: ${item.y} (${item.width}×${item.height}px)</span>
            <button class="di-focus-btn" type="button" title="Locate &amp; spotlight this difference on documents">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
              Spotlight
            </button>
          </div>
          <div class="di-card-desc">${escapeHtml(item.description)}</div>
          <div class="di-snippets">
            <div class="di-snippet-pane di-snippet-a">
              <span class="di-snippet-label">Original (Doc A)</span>
              <img class="di-snippet-img" src="data:image/png;base64,${item.thumb_a_b64}" alt="Snippet A" />
            </div>
            <div class="di-snippet-pane di-snippet-b">
              <span class="di-snippet-label">Modified (Doc B)</span>
              <img class="di-snippet-img" src="data:image/png;base64,${item.thumb_b_b64}" alt="Snippet B" />
            </div>
          </div>
        `;

        card.addEventListener('click', () => {
          focusDifference(item.id);
        });

        els.diCardsGrid.appendChild(card);
      });
    } else {
      els.diCardsGrid.innerHTML = `
        <div style="padding:20px;text-align:center;color:var(--muted);font-size:13.5px;background:var(--bg);border:1px solid var(--border);border-radius:8px;grid-column:1/-1;">
          ✓ No visual differences detected on this page. Both documents match identically at this sensitivity level.
        </div>
      `;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // PIXEL DIFF SUBMODES (Heatmap vs 3-Color)
  // ─────────────────────────────────────────────────────────────
  function setDiffSubmode(mode) {
    state.diffSubmode = mode;

    els.smHeatmap.classList.toggle('active', mode === 'heatmap');
    els.sm3color.classList.toggle('active', mode === '3color');

    if (mode === '3color') {
      els.legAdded.hidden = false;
      els.legMod.hidden = false;
      if (state.diffData && state.diffData.diff_3c_b64) {
        els.imgDiff.src = `data:image/png;base64,${state.diffData.diff_3c_b64}`;
      }
    } else {
      els.legAdded.hidden = true;
      els.legMod.hidden = true;
      if (state.diffData && state.diffData.diff_b64) {
        els.imgDiff.src = `data:image/png;base64,${state.diffData.diff_b64}`;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────
  // VIEW MODE SWITCHING
  // ─────────────────────────────────────────────────────────────
  function switchView(mode) {
    state.activeView = mode;

    els.vtabSBS.classList.toggle('active', mode === 'side-by-side');
    els.vtabDiff.classList.toggle('active', mode === 'diff');
    els.vtabOV.classList.toggle('active', mode === 'overlay');
    els.vtabFlicker.classList.toggle('active', mode === 'flicker');

    els.viewSBS.classList.toggle('active', mode === 'side-by-side');
    els.viewDiff.classList.toggle('active', mode === 'diff');
    els.viewOV.classList.toggle('active', mode === 'overlay');
    els.viewFlicker.classList.toggle('active', mode === 'flicker');

    if (mode === 'overlay') {
      setTimeout(() => setOverlaySplitPosition(50), 50);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // FLASH / FLICKER MODE
  // ─────────────────────────────────────────────────────────────
  function setupFlickerMode() {
    const showVersionB = () => {
      if (!state.diffData) return;
      state.isFlickeringB = true;
      els.imgFlicker.src = `data:image/png;base64,${state.diffData.img_b_b64}`;
      els.flickerBadge.textContent = 'Showing: Document B (Modified)';
      els.flickerBadge.style.color = '#34d399';
      els.btnFlickerHold.classList.add('holding');
    };

    const showVersionA = () => {
      if (!state.diffData) return;
      state.isFlickeringB = false;
      els.imgFlicker.src = `data:image/png;base64,${state.diffData.img_a_b64}`;
      els.flickerBadge.textContent = 'Showing: Document A (Original)';
      els.flickerBadge.style.color = 'white';
      els.btnFlickerHold.classList.remove('holding');
    };

    // Hold Button
    els.btnFlickerHold.addEventListener('mousedown', showVersionB);
    window.addEventListener('mouseup', showVersionA);

    els.btnFlickerHold.addEventListener('touchstart', (e) => {
      e.preventDefault();
      showVersionB();
    }, { passive: false });

    window.addEventListener('touchend', showVersionA);

    // Click on image toggles
    els.imgFlicker.addEventListener('click', () => {
      if (state.isFlickeringB) {
        showVersionA();
      } else {
        showVersionB();
      }
    });
  }

  // ─────────────────────────────────────────────────────────────
  // PAGINATION
  // ─────────────────────────────────────────────────────────────
  function goToPage(newPage) {
    if (newPage < 1 || newPage > state.totalPages || newPage === state.currentPage) {
      return;
    }
    state.currentPage = newPage;
    els.pgInput.value = newPage;
    runComparison(false);
  }

  // ─────────────────────────────────────────────────────────────
  // ZOOM CONTROLS
  // ─────────────────────────────────────────────────────────────
  function adjustZoom(delta) {
    setZoom(Math.max(25, Math.min(300, state.zoomLevel + delta)));
  }

  function setZoom(val) {
    state.zoomLevel = val;
    els.zoomLabel.textContent = `${val}%`;

    const scale = val / 100;
    const transformVal = `scale(${scale})`;

    if (els.wrapA && els.wrapB) {
      els.wrapA.style.transform = transformVal;
      els.wrapB.style.transform = transformVal;
      els.wrapA.style.transformOrigin = 'top center';
      els.wrapB.style.transformOrigin = 'top center';
    } else {
      els.imgA.style.transform = transformVal;
      els.imgB.style.transform = transformVal;
      els.imgA.style.transformOrigin = 'top center';
      els.imgB.style.transformOrigin = 'top center';
    }

    els.imgDiff.style.transform = transformVal;
    els.imgFlicker.style.transform = transformVal;
    els.imgDiff.style.transformOrigin = 'top center';
    els.imgFlicker.style.transformOrigin = 'top center';
  }

  // ─────────────────────────────────────────────────────────────
  // SYNCHRONIZED SCROLLING
  // ─────────────────────────────────────────────────────────────
  function setupSynchronizedScroll() {
    let isSyncingA = false;
    let isSyncingB = false;

    els.scrollA.addEventListener('scroll', () => {
      if (isSyncingA) return;
      isSyncingB = true;
      els.scrollB.scrollTop = els.scrollA.scrollTop;
      els.scrollB.scrollLeft = els.scrollA.scrollLeft;
      requestAnimationFrame(() => (isSyncingB = false));
    });

    els.scrollB.addEventListener('scroll', () => {
      if (isSyncingB) return;
      isSyncingA = true;
      els.scrollA.scrollTop = els.scrollB.scrollTop;
      els.scrollA.scrollLeft = els.scrollB.scrollLeft;
      requestAnimationFrame(() => (isSyncingA = false));
    });
  }

  // ─────────────────────────────────────────────────────────────
  // SPLIT SLIDER OVERLAY
  // ─────────────────────────────────────────────────────────────
  function setOverlaySplitPosition(pct) {
    const clamped = Math.max(0, Math.min(100, pct));
    const width = els.ovContainer.offsetWidth;
    if (!width) return;

    const clipRight = 100 - clamped;
    els.ovClip.style.left = `${clamped}%`;
    els.ovClip.style.right = '0';
    els.ovClip.style.width = `${clipRight}%`;

    els.ovImgB.style.width = `${width}px`;
    els.ovImgB.style.left = `-${(clamped / 100) * width}px`;

    els.ovLine.style.left = `${clamped}%`;
    els.ovHandle.style.left = `${clamped}%`;
  }

  function setupSplitSlider() {
    const handleMove = (clientX) => {
      const rect = els.ovContainer.getBoundingClientRect();
      const x = clientX - rect.left;
      const pct = (x / rect.width) * 100;
      setOverlaySplitPosition(pct);
    };

    els.ovContainer.addEventListener('mousedown', (e) => {
      state.isDraggingSlider = true;
      handleMove(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
      if (state.isDraggingSlider) {
        handleMove(e.clientX);
      }
    });

    window.addEventListener('mouseup', () => {
      state.isDraggingSlider = false;
    });

    els.ovContainer.addEventListener(
      'touchstart',
      (e) => {
        state.isDraggingSlider = true;
        if (e.touches[0]) handleMove(e.touches[0].clientX);
      },
      { passive: true }
    );

    window.addEventListener(
      'touchmove',
      (e) => {
        if (state.isDraggingSlider && e.touches[0]) {
          handleMove(e.touches[0].clientX);
        }
      },
      { passive: true }
    );

    window.addEventListener('touchend', () => {
      state.isDraggingSlider = false;
    });

    if (window.ResizeObserver) {
      new ResizeObserver(() => {
        if (state.activeView === 'overlay') {
          setOverlaySplitPosition(50);
        }
      }).observe(els.ovContainer);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // EXPORT DIFF IMAGE
  // ─────────────────────────────────────────────────────────────
  function exportDiffResult() {
    if (!state.diffData) {
      alert('Please run a comparison first before exporting.');
      return;
    }

    const a = document.createElement('a');
    // Export based on active view
    if (state.activeView === 'diff' && state.diffSubmode === '3color' && state.diffData.diff_3c_b64) {
      a.href = `data:image/png;base64,${state.diffData.diff_3c_b64}`;
    } else {
      a.href = `data:image/png;base64,${state.diffData.diff_b64}`;
    }

    const docA = state.fileA ? state.fileA.name.replace(/\.[^/.]+$/, '') : 'docA';
    const docB = state.fileB ? state.fileB.name.replace(/\.[^/.]+$/, '') : 'docB';
    a.download = `diff_${docA}_vs_${docB}_p${state.currentPage}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // ─────────────────────────────────────────────────────────────
  // KEYBOARD SHORTCUTS
  // ─────────────────────────────────────────────────────────────
  function handleKeyShortcuts(e) {
    if (e.target.tagName === 'INPUT') return;

    if (e.key === '1') switchView('side-by-side');
    if (e.key === '2') switchView('diff');
    if (e.key === '3') switchView('overlay');
    if (e.key === '4') switchView('flicker');

    if (e.key === 'h' || e.key === 'H') toggleHighlights();
    if (e.key === '[' || e.key === '{') navigateDiff(-1);
    if (e.key === ']' || e.key === '}') navigateDiff(1);

    if (e.key === '+' || e.key === '=') adjustZoom(25);
    if (e.key === '-' || e.key === '_') adjustZoom(-25);
    if (e.key === '0') setZoom(100);

    if (state.totalPages > 1) {
      if (e.key === 'ArrowLeft') goToPage(state.currentPage - 1);
      if (e.key === 'ArrowRight') goToPage(state.currentPage + 1);
    }

    // Spacebar to flash in Flicker mode
    if (e.code === 'Space' && state.activeView === 'flicker' && !state.isFlickeringB && state.diffData) {
      e.preventDefault();
      state.isFlickeringB = true;
      els.imgFlicker.src = `data:image/png;base64,${state.diffData.img_b_b64}`;
      els.flickerBadge.textContent = 'Showing: Document B (Modified)';
      els.flickerBadge.style.color = '#34d399';
      els.btnFlickerHold.classList.add('holding');
    }
  }

  function handleKeyUp(e) {
    if (e.code === 'Space' && state.activeView === 'flicker' && state.isFlickeringB && state.diffData) {
      e.preventDefault();
      state.isFlickeringB = false;
      els.imgFlicker.src = `data:image/png;base64,${state.diffData.img_a_b64}`;
      els.flickerBadge.textContent = 'Showing: Document A (Original)';
      els.flickerBadge.style.color = 'white';
      els.btnFlickerHold.classList.remove('holding');
    }
  }

  // Initialize when ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
