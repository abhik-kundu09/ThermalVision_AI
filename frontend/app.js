/**
 * THERMAL VISION — Modern Frontend Client
 * Pinterest-inspired Visual Discovery & Thermal Infrared Enhancement
 */

(() => {
  'use strict';

  // ==========================================================================
  // Application State
  // ==========================================================================
  const state = {
    selectedFile: null,
    isProcessing: false,
    activeViewMode: 'split', // 'split' | 'side-by-side' | 'pipeline'
    activeSampleId: null,
    sliderPercent: 50,
    isDraggingSlider: false,
    currentResults: null,
    sampleDatasets: [],
    activeGalleryFilter: 'all'
  };

  // ==========================================================================
  // DOM Elements Cache
  // ==========================================================================
  const dom = {
    // Navigation & Status
    systemStatusBadge: document.getElementById('systemStatusBadge'),
    statusProviderText: document.getElementById('statusProviderText'),

    // Upload & Dropzone
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    dropzonePrompt: document.getElementById('dropzonePrompt'),
    filePreviewCard: document.getElementById('filePreviewCard'),
    filePreviewImg: document.getElementById('filePreviewImg'),
    previewFileName: document.getElementById('previewFileName'),
    previewFileMeta: document.getElementById('previewFileMeta'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    processBtn: document.getElementById('processBtn'),
    heroUploadBtn: document.getElementById('heroUploadBtn'),

    // Sample Chips & Drawer
    quickSampleChips: document.getElementById('quickSampleChips'),
    tuningToggleBtn: document.getElementById('tuningToggleBtn'),
    tuningChevron: document.getElementById('tuningChevron'),
    tuningContent: document.getElementById('tuningContent'),
    claheClipSlider: document.getElementById('claheClipSlider'),
    claheClipVal: document.getElementById('claheClipVal'),
    bilateralDSlider: document.getElementById('bilateralDSlider'),
    bilateralDVal: document.getElementById('bilateralDVal'),
    unsharpAmountSlider: document.getElementById('unsharpAmountSlider'),
    unsharpAmountVal: document.getElementById('unsharpAmountVal'),
    providerSelect: document.getElementById('providerSelect'),
    resetParamsBtn: document.getElementById('resetParamsBtn'),

    // Viewer & Tabs
    tabSplit: document.getElementById('tabSplit'),
    tabSideBySide: document.getElementById('tabSideBySide'),
    tabPipeline: document.getElementById('tabPipeline'),
    downloadBtn: document.getElementById('downloadBtn'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),

    // Viewport States
    emptyWorkspace: document.getElementById('emptyWorkspace'),
    emptySampleBtn: document.getElementById('emptySampleBtn'),
    processingOverlay: document.getElementById('processingOverlay'),
    processingStepText: document.getElementById('processingStepText'),
    stepUpload: document.getElementById('stepUpload'),
    stepPreprocess: document.getElementById('stepPreprocess'),
    stepColorize: document.getElementById('stepColorize'),
    stepPostprocess: document.getElementById('stepPostprocess'),

    // Comparison Views
    splitSliderView: document.getElementById('splitSliderView'),
    comparisonStage: document.getElementById('comparisonStage'),
    sliderOverlay: document.getElementById('sliderOverlay'),
    sliderHandle: document.getElementById('sliderHandle'),
    imgOriginal: document.getElementById('imgOriginal'),
    imgEnhanced: document.getElementById('imgEnhanced'),

    sideBySideView: document.getElementById('sideBySideView'),
    sideImgOriginal: document.getElementById('sideImgOriginal'),
    sideImgEnhanced: document.getElementById('sideImgEnhanced'),

    pipelineView: document.getElementById('pipelineView'),
    pipelineImg1: document.getElementById('pipelineImg1'),
    pipelineImg2: document.getElementById('pipelineImg2'),
    pipelineImg3: document.getElementById('pipelineImg3'),
    pipelineImg4: document.getElementById('pipelineImg4'),

    // Metrics Bar
    metricsBar: document.getElementById('metricsBar'),
    valLatency: document.getElementById('valLatency'),
    valLatencySub: document.getElementById('valLatencySub'),
    valResolution: document.getElementById('valResolution'),
    valFormat: document.getElementById('valFormat'),
    valSharpnessDelta: document.getElementById('valSharpnessDelta'),
    valTenengradScore: document.getElementById('valTenengradScore'),
    valRmsContrast: document.getElementById('valRmsContrast'),
    valRmsDelta: document.getElementById('valRmsDelta'),
    valEntropy: document.getElementById('valEntropy'),
    valPsnr: document.getElementById('valPsnr'),
    valPsnrNote: document.getElementById('valPsnrNote'),

    // Gallery
    masonryGallery: document.getElementById('masonryGallery'),
    galleryFilters: document.querySelectorAll('.filter-pill'),

    // Technical Accordion
    techAccordionBtn: document.getElementById('techAccordionBtn'),
    techAccordionBody: document.getElementById('techAccordionBody'),
    techChevron: document.getElementById('techChevron'),

    // Lightbox & Toast
    zoomModal: document.getElementById('zoomModal'),
    zoomModalBackdrop: document.getElementById('zoomModalBackdrop'),
    zoomModalImg: document.getElementById('zoomModalImg'),
    zoomCloseBtn: document.getElementById('zoomCloseBtn'),
    zoomCaption: document.getElementById('zoomCaption'),
    toastContainer: document.getElementById('toastContainer')
  };

  // ==========================================================================
  // Initialization
  // ==========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initSplitSlider();
    checkHealth();
    loadSamples();
  });

  function initEventListeners() {
    // Dropzone & File Picker
    if (dom.dropzone) {
      dom.dropzone.addEventListener('click', () => dom.fileInput.click());
      dom.dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          dom.fileInput.click();
        }
      });
      dom.dropzone.addEventListener('dragover', handleDragOver);
      dom.dropzone.addEventListener('dragleave', handleDragLeave);
      dom.dropzone.addEventListener('drop', handleFileDrop);
    }

    if (dom.fileInput) {
      dom.fileInput.addEventListener('change', handleFileSelected);
    }

    if (dom.removeFileBtn) {
      dom.removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
      });
    }

    if (dom.processBtn) {
      dom.processBtn.addEventListener('click', runEnhancementPipeline);
    }

    if (dom.heroUploadBtn) {
      dom.heroUploadBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const card = document.getElementById('uploadCard');
        if (card) {
          card.scrollIntoView({ behavior: 'smooth', block: 'center' });
          dom.fileInput.click();
        }
      });
    }

    if (dom.emptySampleBtn) {
      dom.emptySampleBtn.addEventListener('click', () => {
        if (state.sampleDatasets.length > 0) {
          selectSample(state.sampleDatasets[0], true);
        }
      });
    }

    // Tuning Drawer
    if (dom.tuningToggleBtn) {
      dom.tuningToggleBtn.addEventListener('click', toggleTuningDrawer);
    }

    // Sliders Realtime Output
    bindSliderOutput(dom.claheClipSlider, dom.claheClipVal);
    bindSliderOutput(dom.bilateralDSlider, dom.bilateralDVal);
    bindSliderOutput(dom.unsharpAmountSlider, dom.unsharpAmountVal);

    if (dom.resetParamsBtn) {
      dom.resetParamsBtn.addEventListener('click', resetTuningParameters);
    }

    // Model Switcher Pills — sync with the hidden providerSelect and navbar
    const PROVIDER_LABELS = {
      pytorch_pix2pix: 'Pix2Pix (local)',
      local: 'Colormap (local)'
    };
    const CLOUD_PROVIDERS = new Set(); // no cloud providers

    function updateNavbarProvider(providerKey) {
      if (dom.statusProviderText) {
        dom.statusProviderText.textContent = `Selected: ${PROVIDER_LABELS[providerKey] || providerKey}`;
      }
      if (dom.systemStatusBadge) {
        const dot = dom.systemStatusBadge.querySelector('.status-dot');
        if (dot) {
          // Amber for cloud (needs internet), green for local
          dot.style.background = CLOUD_PROVIDERS.has(providerKey) ? 'var(--warning)' : 'var(--success)';
          dot.style.boxShadow = CLOUD_PROVIDERS.has(providerKey)
            ? '0 0 8px var(--warning)'
            : '0 0 8px var(--success)';
        }
      }
    }

    document.querySelectorAll('.model-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        document.querySelectorAll('.model-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const chosenProvider = pill.dataset.provider;
        if (dom.providerSelect) dom.providerSelect.value = chosenProvider;
        updateNavbarProvider(chosenProvider);
        // No toast here — navbar update is sufficient feedback.
        // A warning will appear automatically if the provider fails during enhance.
      });
    });

    // Keep pills and navbar in sync if providerSelect is changed via tuning drawer
    if (dom.providerSelect) {
      dom.providerSelect.addEventListener('change', () => {
        const val = dom.providerSelect.value;
        document.querySelectorAll('.model-pill').forEach(p => {
          p.classList.toggle('active', p.dataset.provider === val);
        });
        updateNavbarProvider(val);
      });
    }

    // View Modes
    if (dom.tabSplit) dom.tabSplit.addEventListener('click', () => switchViewMode('split'));
    if (dom.tabSideBySide) dom.tabSideBySide.addEventListener('click', () => switchViewMode('side-by-side'));
    if (dom.tabPipeline) dom.tabPipeline.addEventListener('click', () => switchViewMode('pipeline'));

    // Download & Zoom
    if (dom.downloadBtn) dom.downloadBtn.addEventListener('click', downloadEnhancedImage);
    if (dom.fullscreenBtn) dom.fullscreenBtn.addEventListener('click', openZoomModal);
    if (dom.zoomCloseBtn) dom.zoomCloseBtn.addEventListener('click', closeZoomModal);
    if (dom.zoomModalBackdrop) dom.zoomModalBackdrop.addEventListener('click', closeZoomModal);

    // Gallery Filters
    dom.galleryFilters.forEach(pill => {
      pill.addEventListener('click', () => {
        dom.galleryFilters.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        state.activeGalleryFilter = pill.dataset.filter;
        renderMasonryGallery();
      });
    });

    // Technical Accordion
    if (dom.techAccordionBtn) {
      dom.techAccordionBtn.addEventListener('click', toggleTechAccordion);
    }

    // Quick Sample Chips
    document.querySelectorAll('.sample-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const id = chip.dataset.id;
        const filename = chip.dataset.filename;
        const matched = state.sampleDatasets.find(s => s.id === id || s.filename === filename);
        if (matched) {
          selectSample(matched, true);
        } else {
          loadSampleByFilename(filename, chip.textContent.trim(), true);
        }
      });
    });
  }

  function bindSliderOutput(slider, output) {
    if (!slider || !output) return;
    slider.addEventListener('input', () => {
      output.textContent = slider.value;
    });
  }

  // ==========================================================================
  // Health & Sample Loading
  // ==========================================================================
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (dom.statusProviderText) {
          const providerName = data.active_provider === 'local' ? 'Local Mode' : data.active_provider;
          dom.statusProviderText.textContent = `Ready (${providerName})`;
        }
        if (dom.providerSelect && data.active_provider) {
          dom.providerSelect.value = data.active_provider;
        }
      }
    } catch (e) {
      if (dom.statusProviderText) dom.statusProviderText.textContent = 'Offline';
    }
  }

  async function loadSamples() {
    try {
      const res = await fetch('/api/sample-images');
      if (!res.ok) return;
      state.sampleDatasets = await res.json();
      renderMasonryGallery();
    } catch (err) {
      console.warn('Could not load sample datasets:', err);
    }
  }

  // ==========================================================================
  // Masonry Gallery Rendering
  // ==========================================================================
  function renderMasonryGallery() {
    if (!dom.masonryGallery) return;
    dom.masonryGallery.innerHTML = '';

    const filtered = state.sampleDatasets.filter(sample => {
      if (state.activeGalleryFilter === 'all') return true;
      return sample.category.toLowerCase().includes(state.activeGalleryFilter.toLowerCase());
    });

    if (filtered.length === 0) {
      dom.masonryGallery.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-secondary);">
          No sample images found for this category.
        </div>
      `;
      return;
    }

    filtered.forEach(sample => {
      const card = document.createElement('div');
      card.className = 'masonry-card';
      card.innerHTML = `
        <div class="masonry-img-box">
          <img src="/api/sample-images/${sample.filename}" alt="${sample.title}" loading="lazy">
          <div class="masonry-hover-action">
            <button type="button" class="btn-open-workspace">
              <i class="fa-solid fa-arrow-up-right-from-square"></i>
              <span>Open &amp; Enhance</span>
            </button>
          </div>
        </div>
        <div class="masonry-card-content">
          <div class="masonry-card-meta">
            <span class="masonry-category">${sample.category}</span>
            <span class="masonry-dim">${sample.dimensions}</span>
          </div>
          <h3 class="masonry-card-title">${sample.title}</h3>
          <p class="masonry-card-desc">${sample.description}</p>
          <span class="masonry-sensor-tag">${sample.sensor_type}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        selectSample(sample, true);
        const workspace = document.getElementById('workspaceSection');
        if (workspace) {
          workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });

      dom.masonryGallery.appendChild(card);
    });
  }

  // ==========================================================================
  // File Selection & Drag-and-Drop
  // ==========================================================================
  function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dom.dropzone.classList.add('dragover');
  }

  function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dom.dropzone.classList.remove('dragover');
  }

  function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dom.dropzone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }

  function handleFileSelected(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }

  function setSelectedFile(file) {
    state.selectedFile = file;
    state.activeSampleId = null;

    // Reset active sample chips
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));

    dom.processBtn.disabled = false;

    // Read and preview
    const reader = new FileReader();
    reader.onload = (e) => {
      dom.filePreviewImg.src = e.target.result;
      dom.previewFileName.textContent = file.name;

      const img = new Image();
      img.onload = () => {
        const sizeKb = Math.round(file.size / 1024);
        dom.previewFileMeta.textContent = `${img.width} \u00D7 ${img.height} \u2022 ${sizeKb} KB`;
      };
      img.src = e.target.result;

      dom.dropzonePrompt.classList.add('hidden');
      dom.filePreviewCard.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    showToast(`Loaded ${file.name}`, 'info');
  }

  function clearSelectedFile() {
    state.selectedFile = null;
    state.activeSampleId = null;
    dom.fileInput.value = '';
    dom.processBtn.disabled = true;

    dom.dropzonePrompt.classList.remove('hidden');
    dom.filePreviewCard.classList.add('hidden');
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
  }

  async function selectSample(sample, autoRun = false) {
    state.activeSampleId = sample.id;
    document.querySelectorAll('.sample-chip').forEach(c => {
      c.classList.toggle('active', c.dataset.id === sample.id);
    });

    try {
      const res = await fetch(`/api/sample-images/${sample.filename}`);
      if (!res.ok) throw new Error('Sample fetch failed');
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/png' });
      setSelectedFile(file);

      if (autoRun) {
        setTimeout(() => runEnhancementPipeline(), 100);
      }
    } catch (err) {
      showToast(`Could not load sample: ${err.message}`, 'error');
    }
  }

  async function loadSampleByFilename(filename, title, autoRun = false) {
    try {
      const res = await fetch(`/api/sample-images/${filename}`);
      if (!res.ok) throw new Error('Sample fetch failed');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/png' });
      setSelectedFile(file);

      if (autoRun) {
        setTimeout(() => runEnhancementPipeline(), 100);
      }
    } catch (err) {
      showToast(`Could not load sample: ${err.message}`, 'error');
    }
  }

  // ==========================================================================
  // Pipeline Execution & Progress
  // ==========================================================================
  async function runEnhancementPipeline() {
    if (!state.selectedFile || state.isProcessing) return;

    state.isProcessing = true;
    dom.processBtn.disabled = true;
    dom.processBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Processing...</span>';

    // Show processing overlay
    dom.emptyWorkspace.classList.add('hidden');
    dom.processingOverlay.classList.remove('hidden');
    resetProcessingSteps();
    setProcessingStep('upload', 'active', 'Uploading thermal image...');

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    if (dom.providerSelect && dom.providerSelect.value) {
      formData.append('provider', dom.providerSelect.value);
    }
    if (dom.claheClipSlider) formData.append('clahe_clip_limit', dom.claheClipSlider.value);
    if (dom.bilateralDSlider) formData.append('bilateral_d', dom.bilateralDSlider.value);
    if (dom.unsharpAmountSlider) formData.append('unsharp_amount', dom.unsharpAmountSlider.value);

    // Dynamic step animation while in flight
    const timer1 = setTimeout(() => {
      setProcessingStep('upload', 'done');
      setProcessingStep('preprocess', 'active', 'Balancing local contrast & denoising...');
    }, 150);

    const timer2 = setTimeout(() => {
      setProcessingStep('preprocess', 'done');
      setProcessingStep('colorize', 'active', 'Colorizing thermal radiometry...');
    }, 450);

    try {
      const res = await fetch('/api/enhance', {
        method: 'POST',
        body: formData
      });

      clearTimeout(timer1);
      clearTimeout(timer2);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${res.status}`);
      }

      setProcessingStep('colorize', 'done');
      setProcessingStep('postprocess', 'active', 'Refining edge contours & metrics...');

      const data = await res.json();
      state.currentResults = data;

      setTimeout(() => {
        setProcessingStep('postprocess', 'done');
        dom.processingOverlay.classList.add('hidden');
        renderResults(data);
        showToast('Thermal enhancement complete!', 'success');

        if (data.warnings && data.warnings.length > 0) {
          data.warnings.forEach(w => showToast(w, 'warning'));
        }
        if (data.provider_warning) {
          showToast(`⚠ Provider fallback: ${data.provider_warning}`, 'warning');
        }
      }, 200);

    } catch (err) {
      console.error('Enhancement error:', err);
      dom.processingOverlay.classList.add('hidden');
      showToast(err.message || 'Error processing image', 'error');
    } finally {
      state.isProcessing = false;
      dom.processBtn.disabled = false;
      dom.processBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> <span>Enhance Image</span>';
    }
  }

  function setProcessingStep(stepKey, status, text = null) {
    const stepMap = {
      upload: dom.stepUpload,
      preprocess: dom.stepPreprocess,
      colorize: dom.stepColorize,
      postprocess: dom.stepPostprocess
    };

    const stepElem = stepMap[stepKey];
    if (stepElem) {
      stepElem.className = `step-line ${status}`;
    }

    if (text && dom.processingStepText) {
      dom.processingStepText.textContent = text;
    }
  }

  function resetProcessingSteps() {
    [dom.stepUpload, dom.stepPreprocess, dom.stepColorize, dom.stepPostprocess].forEach(step => {
      if (step) step.className = 'step-line';
    });
  }

  // ==========================================================================
  // Result Rendering & Metrics
  // ==========================================================================
  function renderResults(data) {
    // 1. Populate Images
    dom.imgOriginal.src = data.original_image;
    dom.imgEnhanced.src = data.postprocessed_image;

    dom.sideImgOriginal.src = data.original_image;
    dom.sideImgEnhanced.src = data.postprocessed_image;

    dom.pipelineImg1.src = data.original_image;
    dom.pipelineImg2.src = data.preprocessed_image;
    dom.pipelineImg3.src = data.colorized_image;
    dom.pipelineImg4.src = data.postprocessed_image;

    // 2. Enable toolbar buttons & show metrics
    dom.downloadBtn.disabled = false;
    dom.fullscreenBtn.disabled = false;
    dom.metricsBar.classList.remove('hidden');

    // 3. Show active view
    switchViewMode(state.activeViewMode);
    setSliderPercent(50);

    // 4. Update Metrics Cards
    const lat = data.metrics.latency;
    const latencySec = (lat.total_ms / 1000).toFixed(2);
    dom.valLatency.innerHTML = `${latencySec} <span class="unit">s</span>`;
    dom.valLatencySub.textContent = `${lat.total_ms} ms total`;

    const meta = data.metadata;
    dom.valResolution.textContent = `${meta.original_width} \u00D7 ${meta.original_height}`;
    dom.valFormat.textContent = `${meta.bit_depth || '8-bit'} \u2022 ${meta.ai_provider || 'Local'}`;

    // Sharpness gain
    const tIn = data.metrics.tenengrad_sharpness_input;
    const tOut = data.metrics.tenengrad_sharpness_output;
    const delta = tIn > 0 ? (((tOut - tIn) / tIn) * 100).toFixed(0) : 0;
    dom.valSharpnessDelta.textContent = `${delta > 0 ? '+' : ''}${delta}%`;
    dom.valTenengradScore.textContent = `Tenengrad: ${tOut.toLocaleString()}`;

    // Contrast
    dom.valRmsContrast.textContent = data.metrics.rms_contrast_output.toFixed(2);
    const rmsIn = data.metrics.rms_contrast_input;
    const rmsOut = data.metrics.rms_contrast_output;
    const rmsDelta = rmsIn > 0 ? (((rmsOut - rmsIn) / rmsIn) * 100).toFixed(0) : 0;
    dom.valRmsDelta.textContent = `${rmsDelta > 0 ? '+' : ''}${rmsDelta}% dynamic range`;

    // Entropy
    dom.valEntropy.innerHTML = `${data.metrics.shannon_entropy_output.toFixed(1)} <span class="unit">bits</span>`;

    // PSNR / SSIM
    if (data.metrics.has_ground_truth && data.metrics.psnr !== null) {
      dom.valPsnr.textContent = `${data.metrics.psnr.toFixed(1)} dB / ${data.metrics.ssim.toFixed(3)}`;
      dom.valPsnrNote.textContent = 'Paired RGB reference validated';
    } else {
      dom.valPsnr.textContent = 'Reference required';
      dom.valPsnrNote.textContent = 'Requires paired RGB reference';
    }
  }

  // ==========================================================================
  // View Mode Switching
  // ==========================================================================
  function switchViewMode(mode) {
    state.activeViewMode = mode;

    [dom.tabSplit, dom.tabSideBySide, dom.tabPipeline].forEach(t => t.classList.remove('active'));
    [dom.splitSliderView, dom.sideBySideView, dom.pipelineView].forEach(v => v.classList.add('hidden'));

    if (mode === 'split') {
      dom.tabSplit.classList.add('active');
      dom.splitSliderView.classList.remove('hidden');
      setSliderPercent(state.sliderPercent);
    } else if (mode === 'side-by-side') {
      dom.tabSideBySide.classList.add('active');
      dom.sideBySideView.classList.remove('hidden');
    } else if (mode === 'pipeline') {
      dom.tabPipeline.classList.add('active');
      dom.pipelineView.classList.remove('hidden');
    }
  }

  // ==========================================================================
  // Interactive Split Slider Logic
  // ==========================================================================
  function initSplitSlider() {
    if (!dom.comparisonStage || !dom.sliderHandle) return;

    const onPointerDown = (e) => {
      state.isDraggingSlider = true;
      updateSliderFromPointer(e);
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
      window.addEventListener('pointercancel', onPointerUp);
    };

    const onPointerMove = (e) => {
      if (!state.isDraggingSlider) return;
      updateSliderFromPointer(e);
    };

    const onPointerUp = () => {
      state.isDraggingSlider = false;
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
    };

    dom.comparisonStage.addEventListener('pointerdown', onPointerDown);

    // Keyboard accessibility for slider
    dom.sliderHandle.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setSliderPercent(Math.max(0, state.sliderPercent - 5));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setSliderPercent(Math.min(100, state.sliderPercent + 5));
      }
    });
  }

  function updateSliderFromPointer(e) {
    if (!dom.comparisonStage) return;
    const rect = dom.comparisonStage.getBoundingClientRect();
    const clientX = e.clientX;
    const offsetX = clientX - rect.left;
    const percent = Math.min(100, Math.max(0, (offsetX / rect.width) * 100));
    setSliderPercent(percent);
  }

  function setSliderPercent(percent) {
    state.sliderPercent = percent;
    if (dom.sliderOverlay) {
      dom.sliderOverlay.style.clipPath = `polygon(0 0, ${percent}% 0, ${percent}% 100%, 0 100%)`;
    }
    if (dom.sliderHandle) {
      dom.sliderHandle.style.left = `${percent}%`;
      dom.sliderHandle.setAttribute('aria-valuenow', Math.round(percent));
    }
  }

  // ==========================================================================
  // Tuning Drawer & Accordion
  // ==========================================================================
  function toggleTuningDrawer() {
    const isHidden = dom.tuningContent.classList.toggle('hidden');
    dom.tuningToggleBtn.setAttribute('aria-expanded', !isHidden);
    dom.tuningChevron.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
  }

  function resetTuningParameters() {
    if (dom.claheClipSlider) {
      dom.claheClipSlider.value = 3.0;
      dom.claheClipVal.textContent = '3.0';
    }
    if (dom.bilateralDSlider) {
      dom.bilateralDSlider.value = 9;
      dom.bilateralDVal.textContent = '9';
    }
    if (dom.unsharpAmountSlider) {
      dom.unsharpAmountSlider.value = 1.2;
      dom.unsharpAmountVal.textContent = '1.2';
    }
    showToast('Parameters reset to default', 'info');
  }

  function toggleTechAccordion() {
    const isHidden = dom.techAccordionBody.classList.toggle('hidden');
    dom.techAccordionBtn.setAttribute('aria-expanded', !isHidden);
    dom.techChevron.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
  }

  // ==========================================================================
  // Download & Lightbox
  // ==========================================================================
  function downloadEnhancedImage() {
    if (!state.currentResults || !state.currentResults.postprocessed_image) return;
    const a = document.createElement('a');
    a.href = state.currentResults.postprocessed_image;
    const name = (state.selectedFile?.name || 'thermal_enhanced').replace(/\.[^/.]+$/, '');
    a.download = `${name}_enhanced.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast('Download started', 'success');
  }

  function openZoomModal() {
    if (!state.currentResults) return;
    dom.zoomModalImg.src = state.currentResults.postprocessed_image;
    dom.zoomCaption.textContent = state.selectedFile?.name || 'Enhanced Thermal Image';
    dom.zoomModal.classList.remove('hidden');
  }

  function closeZoomModal() {
    dom.zoomModal.classList.add('hidden');
  }

  // ==========================================================================
  // Toast Notifications
  // ==========================================================================
  function showToast(message, type = 'info') {
    if (!dom.toastContainer) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = 'fa-solid fa-circle-info';
    if (type === 'success') icon = 'fa-solid fa-circle-check';
    if (type === 'error') icon = 'fa-solid fa-circle-exclamation';
    if (type === 'warning') icon = 'fa-solid fa-triangle-exclamation';

    toast.innerHTML = `<i class="${icon}"></i><span>${message}</span>`;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

})();
