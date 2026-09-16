/**
 * HandFlow & BlinkFlow Unified Web Client
 * Supports Hand Gestures (Continuous Scroll, Pinch Zoom, Swipes, Poses) and Eye Blinks.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Navigation & Mode Switcher
  const modeHandsBtn = document.getElementById('mode-hands-btn');
  const modeEyesBtn = document.getElementById('mode-eyes-btn');
  const videoImg = document.getElementById('video-stream');
  const masterToggle = document.getElementById('master-toggle');
  const masterStatusLabel = document.getElementById('master-status-label');
  const targetDot = document.getElementById('target-dot');
  const targetText = document.getElementById('target-text');
  const fpsVal = document.getElementById('fps-val');
  const latencyVal = document.getElementById('latency-val');
  const engineStateText = document.getElementById('engine-state-text');
  const triggerFlash = document.getElementById('trigger-flash-fx');

  // Gauges
  const pinchValPct = document.getElementById('pinch-val-pct');
  const pinchGaugeBar = document.getElementById('pinch-gauge-bar');
  const poseValText = document.getElementById('pose-val-text');
  const poseGaugeBar = document.getElementById('pose-gauge-bar');

  // Face Blur Anonymity Button
  const blurFaceBtn = document.getElementById('blur-face-btn');
  const blurIcon = document.getElementById('blur-icon');

  // Audio Toggle
  const audioToggleBtn = document.getElementById('audio-toggle-btn');
  const audioIcon = document.getElementById('audio-icon');

  // Mappings & Tabs
  const mappingContainer = document.getElementById('mapping-list-container');
  const saveMappingsBtn = document.getElementById('save-mappings-btn');
  const saveStatusMsg = document.getElementById('save-status-msg');
  const activityFeedList = document.getElementById('activity-feed-list');
  const historyCount = document.getElementById('history-count');
  const cameraSelect = document.getElementById('camera-select');
  const toggleMeshBtn = document.getElementById('toggle-mesh-btn');
  const presetsContainer = document.getElementById('presets-button-container');

  // Sensitivity Elements
  const sensScrollDeadzone = document.getElementById('sens-scroll-deadzone');
  const sensScrollDeadzoneVal = document.getElementById('sens-scroll-deadzone-val');
  const sensScrollSpeed = document.getElementById('sens-scroll-speed');
  const sensScrollSpeedVal = document.getElementById('sens-scroll-speed-val');
  const sensScrollInverted = document.getElementById('sens-scroll-inverted');
  const sensInputDelay = document.getElementById('sens-input-delay');
  const sensInputDelayVal = document.getElementById('sens-input-delay-val');
  const sensAirMouseEnabled = document.getElementById('sens-air-mouse-enabled');
  const sensAirMouseIdle = document.getElementById('sens-air-mouse-idle');
  const sensAirMouseIdleVal = document.getElementById('sens-air-mouse-idle-val');
  const sensDwellClickEnabled = document.getElementById('sens-dwell-click-enabled');
  const sensDwellClick = document.getElementById('sens-dwell-click');
  const sensDwellClickVal = document.getElementById('sens-dwell-click-val');
  const sensSwipeThresh = document.getElementById('sens-swipe-thresh');
  const sensSwipeThreshVal = document.getElementById('sens-swipe-thresh-val');
  const sensPoseHold = document.getElementById('sens-pose-hold');
  const sensPoseHoldVal = document.getElementById('sens-pose-hold-val');
  const airMousePill = document.getElementById('air-mouse-pill');
  const airMouseDot = document.getElementById('air-mouse-dot');
  const airMouseText = document.getElementById('air-mouse-text');
  const saveTuningBtn = document.getElementById('save-tuning-btn');
  const resetTuningBtn = document.getElementById('reset-tuning-btn');

  // Recorder Modal Elements
  const recorderModal = document.getElementById('recorder-modal');
  const recorderCloseBtn = document.getElementById('recorder-close-btn');
  const recorderCancelBtn = document.getElementById('recorder-cancel-btn');
  const recorderSaveBtn = document.getElementById('recorder-save-btn');
  const recorderPreview = document.getElementById('recorder-preview');

  // App State
  let appConfig = null;
  let ws = null;
  let chart = new WaveformChart('waveform-canvas');
  let audioEnabled = true;
  let activeRecordingGestureKey = null;
  let recordedCombo = "";
  let lastTriggerTime = 0;

  const HAND_GESTURE_META = {
    "SCROLL_UP": { title: "Hand Motion Up", desc: "Swipe / move hand upward", icon: "⬆️", defaultType: "scroll_up" },
    "SCROLL_DOWN": { title: "Hand Motion Down", desc: "Swipe / move hand downward", icon: "⬇️", defaultType: "scroll_down" },
    "SWIPE_LEFT": { title: "Horizontal Swipe Left", desc: "Switch Desktop Left (Responsive flick)", icon: "⬅️", defaultType: "hotkey" },
    "SWIPE_RIGHT": { title: "Horizontal Swipe Right", desc: "Switch Desktop Right (Responsive flick)", icon: "➡️", defaultType: "hotkey" },
    "OPEN_PALM": { title: "Open Palm", desc: "Hold flat open hand", icon: "✋", defaultType: "hotkey" },
    "VICTORY": { title: "Victory / Peace Sign", desc: "Index & middle fingers up (V)", icon: "✌️", defaultType: "hotkey" },
    "THUMB_UP": { title: "Thumbs Up", desc: "Thumb extended upward", icon: "👍", defaultType: "hotkey" },
    "THUMB_DOWN": { title: "Thumbs Down", desc: "Thumb extended downward", icon: "👎", defaultType: "hotkey" },
    "AIR_CLICK": { title: "Air Mouse Click (Fingertip Bend)", desc: "Bend index fingertip down for instant click (terminal / tap)", icon: "🎯", defaultType: "command" },
    "DOUBLE_CLAP": { title: "Double Clap", desc: "Clap hands twice quickly (launches browser)", icon: "👏", defaultType: "command" },
    "SHH_QUIET": { title: "Be Quiet (Finger to Lips)", desc: "Hold index finger to lips (toggle mute)", icon: "🤫", defaultType: "hotkey" },
    "PALM_TO_FIST": { title: "Close Window (Palm ➔ Fist)", desc: "Transition hand from open palm into closed fist to close window", icon: "✊", defaultType: "minimize_window" },
    "FIST_TO_PALM": { title: "Reopen Window (Fist ➔ Palm)", desc: "Transition hand from closed fist into open palm to reopen window", icon: "🖐️", defaultType: "restore_window" },
    "HOLO_EXPAND": { title: "Window Expand (Hands Apart)", desc: "Spread both hands apart horizontally to maximize window", icon: "👐", defaultType: "resize_window_expand" },
    "HOLO_SHRINK": { title: "Window Shrink (Hands Together)", desc: "Bring both hands together to restore window", icon: "🤲", defaultType: "resize_window_shrink" },
    "DUAL_SWIPE_LEFT": { title: "Dual Sweep Left", desc: "Sweep both hands left in unison to switch desktop left", icon: "⏪", defaultType: "switch_desktop_left" },
    "DUAL_SWIPE_RIGHT": { title: "Dual Sweep Right", desc: "Sweep both hands right in unison to switch desktop right", icon: "⏩", defaultType: "switch_desktop_right" },
    "REPULSOR_PUSH": { title: "Palm Push (Show Desktop)", desc: "Thrust flat open palm toward camera to show desktop", icon: "💥", defaultType: "show_desktop" }
  };

  const EYE_GESTURE_META = {
    "DOUBLE_BLINK": { title: "Double Blink", desc: "Two quick blinks in succession", icon: "⚡" },
    "TRIPLE_BLINK": { title: "Triple Blink", desc: "Three deliberate quick blinks", icon: "✨" },
    "LONG_BLINK": { title: "Long Blink (Hold)", desc: "Hold both eyes shut for 500ms+", icon: "⏱️" },
    "LEFT_WINK": { title: "Left Eye Wink", desc: "Quick wink with your left eye", icon: "😉" },
    "RIGHT_WINK": { title: "Right Eye Wink", desc: "Quick wink with your right eye", icon: "😜" },
    "LONG_LEFT_WINK": { title: "Long Left Wink", desc: "Hold left eye shut for 500ms+", icon: "👁️‍🗨️" },
    "LONG_RIGHT_WINK": { title: "Long Right Wink", desc: "Hold right eye shut for 500ms+", icon: "👁️‍🗨️" }
  };

  // -------------------------------------------------------------
  // 1. Initial Load & Setup
  // -------------------------------------------------------------
  fetchConfig();
  fetchCameras();
  setupEventListeners();
  connectWebSocket();

  function fetchConfig() {
    fetch('/api/config')
      .then(res => res.json())
      .then(cfg => {
        appConfig = cfg;
        syncConfigToUI();
      })
      .catch(err => console.error("Error fetching config:", err));
  }

  function fetchCameras() {
    fetch('/api/cameras')
      .then(res => res.json())
      .then(data => {
        cameraSelect.innerHTML = '';
        data.available.forEach(idx => {
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `Camera ${idx} ${idx === data.current ? '(Active)' : ''}`;
          if (idx === data.current) opt.selected = true;
          cameraSelect.appendChild(opt);
        });
      })
      .catch(err => console.error("Error fetching cameras:", err));
  }

  function updateBlurFaceUI(active) {
    if (!blurFaceBtn) return;
    blurFaceBtn.classList.toggle('active', !!active);
    blurFaceBtn.title = active ? "Face Blur Active (Anonymity Shield On)" : "Toggle Face Blur (Anonymity Shield)";
    if (blurIcon) blurIcon.textContent = active ? '🕶️' : '👤';
  }

  function syncConfigToUI() {
    if (!appConfig) return;

    // Mode
    updateModeUI(appConfig.mode);

    // Face Blur Anonymity
    if (appConfig.blur_face !== undefined) {
      updateBlurFaceUI(appConfig.blur_face);
    }

    // Master switch
    masterToggle.checked = appConfig.active;
    updateMasterStatusUI(appConfig.active);

    // Audio
    audioEnabled = appConfig.audio_feedback;
    window.soundSynth.toggle(audioEnabled);
    audioIcon.textContent = audioEnabled ? '🔊' : '🔇';

    // HUD Mesh
    toggleMeshBtn.classList.toggle('active', appConfig.draw_mesh);

    // Sensitivity
    const hSens = appConfig.hand_sensitivity;
    if (hSens) {
      sensScrollDeadzone.value = hSens.scroll_deadzone;
      sensScrollDeadzoneVal.textContent = hSens.scroll_deadzone.toFixed(3);

      sensScrollSpeed.value = hSens.scroll_speed;
      sensScrollSpeedVal.textContent = hSens.scroll_speed;

      sensScrollInverted.checked = hSens.scroll_inverted;

      sensInputDelay.value = hSens.input_delay_ms !== undefined ? hSens.input_delay_ms : 250;
      sensInputDelayVal.textContent = `${sensInputDelay.value}ms`;

      if (sensAirMouseEnabled) sensAirMouseEnabled.checked = hSens.air_mouse_enabled !== false;
      if (sensAirMouseIdle) {
        sensAirMouseIdle.value = hSens.air_mouse_idle_sec || 2.0;
        sensAirMouseIdleVal.textContent = `${parseFloat(sensAirMouseIdle.value).toFixed(1)}s`;
      }
      if (sensDwellClickEnabled) sensDwellClickEnabled.checked = hSens.dwell_click_enabled === true;
      if (sensDwellClick) {
        sensDwellClick.value = hSens.dwell_click_ms || 700;
        sensDwellClickVal.textContent = `${sensDwellClick.value}ms`;
      }

      sensSwipeThresh.value = hSens.swipe_velocity_thresh;
      sensSwipeThreshVal.textContent = hSens.swipe_velocity_thresh.toFixed(2);

      sensPoseHold.value = hSens.pose_hold_ms;
      sensPoseHoldVal.textContent = `${hSens.pose_hold_ms}ms`;
    }

    renderMappings();
  }

  function updateModeUI(mode) {
    if (mode === 'hands') {
      modeHandsBtn.classList.add('active');
      modeEyesBtn.classList.remove('active');
      document.querySelector('.brand-title').innerHTML = 'HAND<span>FLOW</span>';
      document.querySelector('.brand-tag').textContent = 'Touchless Vision Gesture Controller';
      targetText.textContent = 'SEARCHING HAND';
      document.getElementById('chart-title-text').textContent = 'Real-Time Hand Motion & Activity Waveform';
      document.getElementById('chart-legend-container').innerHTML = `
        <span class="legend-item"><span class="dot left-eye-dot"></span>Activity</span>
        <span class="legend-item"><span class="dot right-eye-dot"></span>Motion Y</span>
      `;
      presetsContainer.innerHTML = `
        <button class="btn btn-pill active" data-preset="browsing">🌐 Browsing & Scroll</button>
        <button class="btn btn-pill" data-preset="media">🎵 Media & Volume</button>
        <button class="btn btn-pill" data-preset="presentation">📽️ Presentation Slides</button>
      `;
      setupPresetButtons();
    } else {
      modeEyesBtn.classList.add('active');
      modeHandsBtn.classList.remove('active');
      document.querySelector('.brand-title').innerHTML = 'BLINK<span>FLOW</span>';
      document.querySelector('.brand-tag').textContent = 'Neural Vision Shortcut Controller';
      targetText.textContent = 'SEARCHING FACE';
      document.getElementById('chart-title-text').textContent = 'Real-Time Eye Openness Waveform';
      document.getElementById('chart-legend-container').innerHTML = `
        <span class="legend-item"><span class="dot left-eye-dot"></span>Left</span>
        <span class="legend-item"><span class="dot right-eye-dot"></span>Right</span>
      `;
      presetsContainer.innerHTML = `
        <button class="btn btn-pill" data-preset="reading">📖 Reading</button>
        <button class="btn btn-pill" data-preset="browser">🌐 Browser</button>
        <button class="btn btn-pill" data-preset="media">🎵 Media</button>
      `;
      setupPresetButtons();
    }
  }

  function renderMappings() {
    if (!appConfig) return;
    mappingContainer.innerHTML = '';

    const isHands = appConfig.mode === 'hands';
    const mappings = isHands ? appConfig.hand_mappings : appConfig.mappings;
    const metaMap = isHands ? HAND_GESTURE_META : EYE_GESTURE_META;

    Object.keys(metaMap).forEach(key => {
      const mapping = mappings[key] || {
        type: metaMap[key].defaultType || "hotkey",
        value: "",
        description: "",
        enabled: false
      };

      const meta = metaMap[key];

      const row = document.createElement('div');
      row.className = 'mapping-row';
      row.dataset.gesture = key;

      const typeOptions = isHands ? `
        <option value="hotkey" ${mapping.type === 'hotkey' ? 'selected' : ''}>Hotkey</option>
        <option value="command" ${mapping.type === 'command' ? 'selected' : ''}>Command</option>
        <option value="minimize_window" ${mapping.type === 'minimize_window' ? 'selected' : ''}>Minimize Window</option>
        <option value="restore_window" ${mapping.type === 'restore_window' ? 'selected' : ''}>Restore Window</option>
        <option value="resize_window_expand" ${mapping.type === 'resize_window_expand' ? 'selected' : ''}>Maximize / Expand Window</option>
        <option value="resize_window_shrink" ${mapping.type === 'resize_window_shrink' ? 'selected' : ''}>Restore / Shrink Window</option>
        <option value="show_desktop" ${mapping.type === 'show_desktop' ? 'selected' : ''}>Show Desktop</option>
        <option value="switch_desktop_left" ${mapping.type === 'switch_desktop_left' ? 'selected' : ''}>Switch Desktop Left</option>
        <option value="switch_desktop_right" ${mapping.type === 'switch_desktop_right' ? 'selected' : ''}>Switch Desktop Right</option>
        <option value="click" ${mapping.type === 'click' ? 'selected' : ''}>Mouse Click</option>
        <option value="scroll_up" ${mapping.type === 'scroll_up' ? 'selected' : ''}>Scroll Up</option>
        <option value="scroll_down" ${mapping.type === 'scroll_down' ? 'selected' : ''}>Scroll Down</option>
      ` : `
        <option value="hotkey" ${mapping.type === 'hotkey' ? 'selected' : ''}>Hotkey</option>
        <option value="command" ${mapping.type === 'command' ? 'selected' : ''}>Command</option>
      `;

      row.innerHTML = `
        <div class="gesture-info">
          <div class="gesture-title">${meta.icon} ${meta.title}</div>
          <div class="gesture-desc">${meta.desc}</div>
        </div>

        <select class="mapping-type-select" data-key="${key}">
          ${typeOptions}
        </select>

        <input type="text" class="mapping-value-input" data-key="${key}" value="${escapeHtml(mapping.value)}" placeholder="Key combo, command or steps">

        <div class="mapping-row-actions">
          <button class="btn btn-xs record-key-btn" data-key="${key}" title="Record Key Combo">Record</button>
          <button class="btn btn-xs test-action-btn" data-key="${key}" title="Test Action Now">Test</button>
        </div>

        <label class="switch" title="Enable/Disable gesture">
          <input type="checkbox" class="mapping-toggle" data-key="${key}" ${mapping.enabled ? 'checked' : ''}>
          <span class="slider round"></span>
        </label>
      `;

      mappingContainer.appendChild(row);
    });

    // Wire up row buttons
    mappingContainer.querySelectorAll('.record-key-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        openRecorderModal(e.target.dataset.key);
      });
    });

    mappingContainer.querySelectorAll('.test-action-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        testAction(e.target.dataset.key);
      });
    });
  }

  // -------------------------------------------------------------
  // 2. WebSocket Telemetry Stream
  // -------------------------------------------------------------
  function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.frame) {
          videoImg.src = `data:image/jpeg;base64,${msg.frame}`;
        }
        updateTelemetry(msg);
      } catch (e) {
        console.error("Error parsing WS telemetry:", e);
      }
    };

    ws.onclose = () => {
      setTimeout(connectWebSocket, 1500);
    };
  }

  function updateTelemetry(msg) {
    if (msg.fps !== undefined) fpsVal.textContent = msg.fps.toFixed(1);
    if (msg.latency_ms !== undefined) latencyVal.textContent = `${Math.round(msg.latency_ms)}ms`;
    if (msg.blur_face !== undefined) updateBlurFaceUI(msg.blur_face);

    if (msg.mode === 'hands') {
      // Hand tracking telemetry
      if (msg.hand_detected) {
        targetDot.className = 'status-dot active';
        if (msg.num_hands >= 2) {
          targetText.textContent = `DUAL HANDS TRACKED (${msg.num_hands})`;
        } else {
          targetText.textContent = `HAND TRACKED (${msg.num_hands})`;
        }
      } else {
        targetDot.className = 'status-dot warning';
        targetText.textContent = 'SEARCHING HAND';
      }

      // Hand Activity, Air Mouse & Pose Gauges
      if (msg.air_mouse_active) {
        if (airMouseDot) airMouseDot.className = 'status-dot active';
        if (airMouseText) airMouseText.textContent = 'AIR MOUSE: ACTIVE';

        const dwellPct = Math.round((msg.dwell_progress || 0) * 100);
        pinchGaugeBar.style.width = `${dwellPct}%`;
        pinchGaugeBar.style.background = 'var(--accent-amber, #f59e0b)';
        pinchValPct.textContent = dwellPct > 5 ? `DWELL CLICK: ${dwellPct}%` : 'POINTER TRACKING';
      } else if ((msg.idle_progress || 0) > 0.05) {
        if (airMouseDot) airMouseDot.className = 'status-dot warning';
        const chargePct = Math.round((msg.idle_progress || 0) * 100);
        if (airMouseText) airMouseText.textContent = `CHARGING: ${chargePct}%`;

        pinchGaugeBar.style.width = `${chargePct}%`;
        pinchGaugeBar.style.background = 'var(--accent-purple, #a855f7)';
        pinchValPct.textContent = `AIR MOUSE: ${chargePct}%`;
      } else {
        if (airMouseDot) airMouseDot.className = 'status-dot';
        if (airMouseText) airMouseText.textContent = 'AIR MOUSE: IDLE';

        const isCooldown = msg.engine_state && msg.engine_state.includes("DELAY");
        if (isCooldown) {
          pinchGaugeBar.style.width = '100%';
          pinchGaugeBar.style.background = 'var(--accent-cyan, #06b6d4)';
          pinchValPct.textContent = 'COOLDOWN (250ms)';
        } else {
          const actPct = msg.hand_detected ? 100 : 0;
          pinchGaugeBar.style.width = `${actPct}%`;
          pinchGaugeBar.style.background = '';
          pinchValPct.textContent = msg.hand_detected ? 'ACTIVE / READY' : 'NO HAND';
        }
      }
      
      const pose = msg.pose || 'None';
      poseValText.textContent = pose;

      // Push into chart
      chart.pushSample(msg.hand_detected ? 0.8 : 0.0, msg.air_mouse_active ? 0.9 : (msg.idle_progress || 0.2));

    } else {
      // Eye tracking telemetry
      if (msg.face_detected) {
        targetDot.className = 'status-dot active';
        targetText.textContent = 'FACE TRACKED';
      } else {
        targetDot.className = 'status-dot warning';
        targetText.textContent = 'SEARCHING FACE';
      }

      const leftPct = Math.round((msg.left_blink || 0) * 100);
      pinchGaugeBar.style.width = `${leftPct}%`;
      pinchValPct.textContent = `${leftPct}%`;
      poseValText.textContent = `L: ${leftPct}% | R: ${Math.round((msg.right_blink || 0) * 100)}%`;

      chart.pushSample(msg.left_blink || 0, msg.right_blink || 0);
    }

    // Engine State & Flash
    if (msg.engine_state) {
      engineStateText.textContent = msg.engine_state;
      if (msg.engine_state.includes("SCROLL") || msg.engine_state.includes("SWIPE") || msg.engine_state.includes("TRIGGER") || msg.engine_state.includes("HOLO") || msg.engine_state.includes("REPULSOR") || msg.engine_state.includes("WINDOW")) {
        triggerFlashEffect(msg.engine_state);
      }
    }

    if (msg.recent_actions && msg.recent_actions.length > 0) {
      renderActivityFeed(msg.recent_actions);
    }
  }

  function triggerFlashEffect(engineState = "") {
    const now = Date.now();
    if (now - lastTriggerTime > 250) {
      lastTriggerTime = now;
      triggerFlash.classList.add('flash');
      if (engineState.includes("EXPAND")) {
        window.soundSynth.playHoloExpand();
      } else if (engineState.includes("SHRINK")) {
        window.soundSynth.playHoloShrink();
      } else if (engineState.includes("REPULSOR")) {
        window.soundSynth.playRepulsor();
      } else {
        window.soundSynth.playTrigger();
      }
      setTimeout(() => triggerFlash.classList.remove('flash'), 220);
    }
  }

  function renderActivityFeed(actions) {
    historyCount.textContent = actions.length;
    activityFeedList.innerHTML = '';
    actions.forEach(act => {
      const item = document.createElement('div');
      item.className = `feed-item ${act.success ? '' : 'error'}`;
      item.innerHTML = `
        <div class="feed-meta">
          <span class="feed-time">${act.time_str || ''}</span>
          <span class="feed-gesture">${act.gesture}</span>
          <span class="feed-action">→ ${escapeHtml(act.type)}: ${escapeHtml(act.value)}</span>
        </div>
        <span class="feed-status ${act.success ? 'ok' : 'fail'}">${act.success ? 'OK' : 'ERR'}</span>
      `;
      activityFeedList.appendChild(item);
    });
  }

  // -------------------------------------------------------------
  // 3. Actions & Presets
  // -------------------------------------------------------------
  function testAction(gestureKey) {
    const row = document.querySelector(`.mapping-row[data-gesture="${gestureKey}"]`);
    if (!row) return;

    const type = row.querySelector('.mapping-type-select').value;
    const value = row.querySelector('.mapping-value-input').value;

    triggerFlashEffect();

    fetch('/api/test-action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gesture: gestureKey, type: type, value: value })
    });
  }

  function saveCurrentMappings() {
    if (!appConfig) return;

    const updated = {};
    document.querySelectorAll('.mapping-row').forEach(row => {
      const key = row.dataset.gesture;
      const type = row.querySelector('.mapping-type-select').value;
      const value = row.querySelector('.mapping-value-input').value.trim();
      const enabled = row.querySelector('.mapping-toggle').checked;

      updated[key] = {
        type: type,
        value: value,
        description: value,
        enabled: enabled
      };
    });

    const isHands = appConfig.mode === 'hands';
    const payload = isHands ? { hand_mappings: updated } : { mappings: updated };

    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      appConfig = data.config;
      saveStatusMsg.textContent = "✓ Saved Successfully";
      setTimeout(() => { saveStatusMsg.textContent = ""; }, 2500);
    });
  }

  function setupPresetButtons() {
    document.querySelectorAll('.btn-pill[data-preset]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.btn-pill[data-preset]').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const preset = e.target.dataset.preset;
        fetch('/api/preset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preset: preset })
        })
        .then(res => res.json())
        .then(data => {
          if (appConfig.mode === 'hands') appConfig.hand_mappings = data.mappings;
          else appConfig.mappings = data.mappings;
          renderMappings();
          window.soundSynth.playTrigger();
        });
      });
    });
  }

  // -------------------------------------------------------------
  // 4. Sensitivity Sliders & Mode
  // -------------------------------------------------------------
  function saveSensitivityTuning() {
    const updates = {
      hand_sensitivity: {
        scroll_deadzone: parseFloat(sensScrollDeadzone.value),
        scroll_speed: parseInt(sensScrollSpeed.value),
        scroll_inverted: sensScrollInverted.checked,
        input_delay_ms: parseInt(sensInputDelay.value),
        air_mouse_enabled: sensAirMouseEnabled ? sensAirMouseEnabled.checked : true,
        air_mouse_idle_sec: sensAirMouseIdle ? parseFloat(sensAirMouseIdle.value) : 2.0,
        dwell_click_enabled: sensDwellClickEnabled ? sensDwellClickEnabled.checked : false,
        dwell_click_ms: sensDwellClick ? parseInt(sensDwellClick.value) : 700,
        swipe_velocity_thresh: parseFloat(sensSwipeThresh.value),
        swipe_min_dist: 0.08,
        pose_hold_ms: parseInt(sensPoseHold.value)
      }
    };

    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    })
    .then(res => res.json())
    .then(data => {
      appConfig = data.config;
      syncConfigToUI();
      alert("Motion sensitivity updated!");
    });
  }

  function switchMode(newMode) {
    fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode })
    })
    .then(res => res.json())
    .then(data => {
      appConfig.mode = data.mode;
      updateModeUI(data.mode);
      renderMappings();
      window.soundSynth.playTrigger();
    });
  }

  // -------------------------------------------------------------
  // 5. Hotkey Recorder Modal
  // -------------------------------------------------------------
  function openRecorderModal(gestureKey) {
    activeRecordingGestureKey = gestureKey;
    recordedCombo = "";
    recorderPreview.textContent = "Press key combination now...";
    recorderModal.style.display = 'flex';
    window.addEventListener('keydown', handleRecorderKeyDown);
  }

  function handleRecorderKeyDown(e) {
    e.preventDefault();
    e.stopPropagation();

    const parts = [];
    if (e.ctrlKey) parts.push("ctrl");
    if (e.altKey) parts.push("alt");
    if (e.shiftKey) parts.push("shift");
    if (e.metaKey) parts.push("super");

    const key = e.key;
    if (!["Control", "Alt", "Shift", "Meta"].includes(key)) {
      if (key === " ") parts.push("space");
      else if (key === "ArrowUp") parts.push("Up");
      else if (key === "ArrowDown") parts.push("Down");
      else if (key === "ArrowLeft") parts.push("Left");
      else if (key === "ArrowRight") parts.push("Right");
      else if (key === "PageDown") parts.push("Page_Down");
      else if (key === "PageUp") parts.push("Page_Up");
      else if (key === "Escape") parts.push("Escape");
      else if (key === "Enter") parts.push("Return");
      else parts.push(key);

      recordedCombo = parts.join("+");
      recorderPreview.textContent = recordedCombo;
    }
  }

  function closeRecorderModal() {
    window.removeEventListener('keydown', handleRecorderKeyDown);
    recorderModal.style.display = 'none';
    activeRecordingGestureKey = null;
    recordedCombo = "";
  }

  function saveRecordedHotkey() {
    if (activeRecordingGestureKey && recordedCombo) {
      const row = document.querySelector(`.mapping-row[data-gesture="${activeRecordingGestureKey}"]`);
      if (row) {
        row.querySelector('.mapping-type-select').value = "hotkey";
        row.querySelector('.mapping-value-input').value = recordedCombo;
      }
      saveCurrentMappings();
    }
    closeRecorderModal();
  }

  // -------------------------------------------------------------
  // 6. Event Listeners Setup
  // -------------------------------------------------------------
  function setupEventListeners() {
    // Mode Switcher Buttons
    modeHandsBtn.addEventListener('click', () => switchMode('hands'));
    modeEyesBtn.addEventListener('click', () => switchMode('eyes'));

    // Master switch
    masterToggle.addEventListener('change', () => {
      fetch('/api/toggle-active', { method: 'POST' })
        .then(res => res.json())
        .then(data => updateMasterStatusUI(data.active));
    });

    // Face Blur Anonymity toggle
    if (blurFaceBtn) {
      blurFaceBtn.addEventListener('click', () => {
        fetch('/api/blur-face', { method: 'POST' })
          .then(res => res.json())
          .then(data => {
            if (appConfig) appConfig.blur_face = data.blur_face;
            updateBlurFaceUI(data.blur_face);
            window.soundSynth.playTrigger();
          })
          .catch(err => console.error("Error toggling face blur:", err));
      });
    }

    // Audio toggle
    audioToggleBtn.addEventListener('click', () => {
      audioEnabled = !audioEnabled;
      window.soundSynth.toggle(audioEnabled);
      audioIcon.textContent = audioEnabled ? '🔊' : '🔇';
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_feedback: audioEnabled })
      });
    });

    // AR Mesh toggle
    toggleMeshBtn.addEventListener('click', () => {
      const active = toggleMeshBtn.classList.toggle('active');
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ draw_mesh: active })
      });
    });

    // Camera switch
    cameraSelect.addEventListener('change', (e) => {
      const newIdx = parseInt(e.target.value);
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: newIdx })
      });
    });

    // Save mappings
    saveMappingsBtn.addEventListener('click', saveCurrentMappings);

    // Hand Sensitivity Slider Listeners
    sensScrollDeadzone.addEventListener('input', (e) => {
      sensScrollDeadzoneVal.textContent = parseFloat(e.target.value).toFixed(3);
    });

    sensScrollSpeed.addEventListener('input', (e) => {
      sensScrollSpeedVal.textContent = e.target.value;
    });

    sensInputDelay.addEventListener('input', (e) => {
      sensInputDelayVal.textContent = `${e.target.value}ms`;
    });

    if (sensAirMouseIdle) {
      sensAirMouseIdle.addEventListener('input', (e) => {
        sensAirMouseIdleVal.textContent = `${parseFloat(e.target.value).toFixed(1)}s`;
      });
    }

    if (sensDwellClick) {
      sensDwellClick.addEventListener('input', (e) => {
        sensDwellClickVal.textContent = `${e.target.value}ms`;
      });
    }

    sensSwipeThresh.addEventListener('input', (e) => {
      sensSwipeThreshVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    sensPoseHold.addEventListener('input', (e) => {
      sensPoseHoldVal.textContent = `${e.target.value}ms`;
    });

    saveTuningBtn.addEventListener('click', saveSensitivityTuning);

    resetTuningBtn.addEventListener('click', () => {
      sensScrollDeadzone.value = 0.038;
      sensScrollDeadzone.dispatchEvent(new Event('input'));
      sensScrollSpeed.value = 3;
      sensScrollSpeed.dispatchEvent(new Event('input'));
      sensScrollInverted.checked = false;
      sensInputDelay.value = 250;
      sensInputDelay.dispatchEvent(new Event('input'));
      if (sensAirMouseEnabled) sensAirMouseEnabled.checked = true;
      if (sensAirMouseIdle) {
        sensAirMouseIdle.value = 2.0;
        sensAirMouseIdle.dispatchEvent(new Event('input'));
      }
      if (sensDwellClickEnabled) sensDwellClickEnabled.checked = false;
      if (sensDwellClick) {
        sensDwellClick.value = 700;
        sensDwellClick.dispatchEvent(new Event('input'));
      }
      sensSwipeThresh.value = 0.45;
      sensSwipeThresh.dispatchEvent(new Event('input'));
      sensPoseHold.value = 350;
      sensPoseHold.dispatchEvent(new Event('input'));
      saveSensitivityTuning();
    });

    // Tab Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        e.target.classList.add('active');
        const pane = document.getElementById(e.target.dataset.tab);
        if (pane) pane.classList.add('active');
      });
    });

    // Recorder modal buttons
    recorderCloseBtn.addEventListener('click', closeRecorderModal);
    recorderCancelBtn.addEventListener('click', closeRecorderModal);
    recorderSaveBtn.addEventListener('click', saveRecordedHotkey);
  }

  function updateMasterStatusUI(active) {
    if (active) {
      masterStatusLabel.textContent = "ACTIVE";
      masterStatusLabel.className = "switch-label";
    } else {
      masterStatusLabel.textContent = "PAUSED";
      masterStatusLabel.className = "switch-label paused";
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
});
