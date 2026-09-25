"use strict";

// Web page for the btd API. Images go to POST /v1/predict; the returned polygons are drawn on a canvas, and the
// threshold slider filters detections in the browser (the server already returns everything above the model's
// operating threshold), so moving it never needs another request.
(() => {
  // Okabe-Ito, the same colours as the server-side overlay (btd.inference.visualize).
  const COLORS = { glioma: "#E69F00", meningioma: "#56B4E9", pituitary: "#009E73" };
  const FALLBACK_COLOR = "#CC79A7";
  const NAMES = { glioma: "Glioma", meningioma: "Meningioma", pituitary: "Pituitary", no_tumor: "No tumour" };

  const $ = (id) => document.getElementById(id);
  const state = {
    image: null,
    result: null,
    expert: null,
    wantExpert: false, // remembered across slices, even ones without an expert mask
    apiKey: null,
    busy: false,
    pending: 0,
    retry: null,
  };

  const colorOf = (cls) => COLORS[cls] || FALLBACK_COLOR;
  const nameOf = (cls) => NAMES[cls] || cls;
  const providerName = (p) => String(p || "").replace(/ExecutionProvider$/, "").toLowerCase() || "unknown";
  const threshold = () => Number($("threshold").value);

  // ---------------------------------------------------------------------------------------------- model + samples
  async function loadModel() {
    try {
      const r = await fetch("/v1/model");
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
      $("model-badge").textContent = `${body.name} · ${body.precision} · ${providerName(body.providers[0])}`;
      const slider = $("threshold");
      slider.min = String(body.conf_threshold);
      slider.value = String(body.conf_threshold);
      updateThresholdOut();
    } catch (err) {
      $("model-badge").textContent = "model not loaded";
      showError(`The model isn't available right now (${err.message}).`);
    }
  }

  async function loadSamples() {
    let samples = [];
    try {
      const r = await fetch("/ui/samples/samples.json");
      if (r.ok) ({ samples } = await r.json());
    } catch {
      return []; // samples are optional
    }
    return samples.map((sample) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn";
      button.textContent = sample.title;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => runSample(sample, button));
      $("samples").append(button);
      return { sample, button };
    });
  }

  // Shareable links such as /?sample=glioma&expert=1&threshold=0.1 (also used for the README screenshots).
  function applyLinkParams(samples) {
    const params = new URLSearchParams(window.location.search);
    const t = Number(params.get("threshold"));
    const slider = $("threshold");
    if (params.has("threshold") && t >= Number(slider.min) && t <= Number(slider.max)) {
      slider.value = String(t);
      updateThresholdOut();
    }
    if (params.get("expert") === "1") state.wantExpert = true;
    const wanted = (params.get("sample") || "").toLowerCase();
    const match = samples.find(({ sample }) => [sample.label, sample.title.toLowerCase()].includes(wanted));
    if (match) runSample(match.sample, match.button);
  }

  function setPressed(active) {
    for (const b of $("samples").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === active));
  }

  async function runSample(sample, button) {
    setPressed(button);
    try {
      const r = await fetch(`/ui/samples/${encodeURIComponent(sample.file)}`);
      if (!r.ok) throw new Error();
      const blob = await r.blob();
      analyse(new File([blob], sample.file, { type: blob.type || "image/jpeg" }), sample.expert_polygons);
    } catch {
      showError("Couldn't load that sample. Reload the page and try again.");
    }
  }

  // ---------------------------------------------------------------------------------------------- analysis
  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("decode"));
      };
      img.src = url;
    });
  }

  function errorText(status, body) {
    const detail = body && body.detail;
    if (typeof detail === "string") return /[.!?]$/.test(detail) ? detail : `${detail}.`;
    if (status === 413) return "That file is too large.";
    return `The server couldn't analyse that image (HTTP ${status}).`;
  }

  async function analyse(file, expert = null) {
    hideError();
    let img;
    try {
      img = await loadImage(file);
    } catch {
      showError("That file can't be shown here. Use a JPEG, PNG, BMP, or WebP image.");
      return;
    }
    const ticket = ++state.pending; // a newer image wins if the user switches mid-request
    state.image = img;
    state.expert = expert && expert.length ? expert : null;
    state.result = null;
    $("viewer-empty").hidden = true;
    setBusy(true);
    render();

    try {
      const form = new FormData();
      form.append("file", file, file.name || "slice.png");
      const headers = state.apiKey ? { "X-API-Key": state.apiKey } : {};
      const r = await fetch("/v1/predict", { method: "POST", body: form, headers });
      const body = await r.json().catch(() => ({}));
      if (ticket !== state.pending) return;
      if (r.status === 401) {
        state.retry = () => analyse(file, expert);
        $("key-form").hidden = false;
        $("api-key").focus();
        showError("This server needs an API key. Enter it below.");
      } else if (!r.ok) {
        showError(errorText(r.status, body));
      } else {
        state.result = body;
      }
    } catch {
      if (ticket === state.pending) showError("Couldn't reach the server. Check it's running and try again.");
    } finally {
      if (ticket === state.pending) {
        setBusy(false);
        render();
      }
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    $("viewer-busy").hidden = !busy;
  }

  function showError(message) {
    $("error").textContent = message;
    $("error").hidden = false;
  }

  function hideError() {
    $("error").hidden = true;
  }

  // ---------------------------------------------------------------------------------------------- drawing
  function visibleDetections() {
    if (!state.result) return [];
    const t = threshold();
    return state.result.detections
      .filter((d) => d.confidence >= t - 1e-9)
      .sort((a, b) => b.confidence - a.confidence);
  }

  function tracePolygons(ctx, polygons, tx, ty) {
    ctx.beginPath();
    for (const poly of polygons) {
      poly.forEach(([x, y], i) => (i ? ctx.lineTo(tx(x), ty(y)) : ctx.moveTo(tx(x), ty(y))));
      ctx.closePath();
    }
  }

  function drawMask(ctx, d, tx, ty) {
    if (!d.polygons || !d.polygons.length) return;
    const color = colorOf(d.class_name);
    tracePolygons(ctx, d.polygons, tx, ty);
    if ($("show-mask").checked) {
      ctx.globalAlpha = 0.42;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    if ($("show-outline").checked) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }

  const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

  // `placed` holds the label rectangles already drawn; a label that would cover one moves below its box instead.
  function drawBox(ctx, d, tx, ty) {
    const x1 = tx(d.box.x1);
    const y1 = ty(d.box.y1);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = colorOf(d.class_name);
    ctx.strokeRect(x1, y1, tx(d.box.x2) - x1, ty(d.box.y2) - y1);
  }

  function drawLabel(ctx, d, tx, ty, placed, canvasHeight) {
    const color = colorOf(d.class_name);
    const x1 = tx(d.box.x1);
    const y1 = ty(d.box.y1);
    const y2 = ty(d.box.y2);
    const text = `${d.class_name} ${d.confidence.toFixed(2)}`;
    ctx.font = "600 12px system-ui, sans-serif";
    const w = ctx.measureText(text).width + 10;
    const h = 18;
    const candidates = [y1 - h, y2, y1].filter((y) => y >= 0 && y + h <= canvasHeight);
    const top = candidates.find((y) => !placed.some((r) => overlaps(r, { x: x1, y, w, h }))) ?? candidates[0] ?? y1;
    placed.push({ x: x1, y: top, w, h });
    ctx.fillStyle = color;
    ctx.fillRect(x1, top, w, h);
    ctx.fillStyle = "#000";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x1 + 5, top + h / 2 + 0.5);
  }

  function render() {
    const viewer = $("viewer");
    const canvas = $("canvas");
    const dpr = window.devicePixelRatio || 1;
    const w = viewer.clientWidth;
    const h = viewer.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);

    const img = state.image;
    if (img) {
      const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
      const dw = img.naturalWidth * scale;
      const dh = img.naturalHeight * scale;
      const ox = (w - dw) / 2;
      const oy = (h - dh) / 2;
      ctx.drawImage(img, ox, oy, dw, dh);
      // Polygons are in the pixel grid the server decoded; map them onto the displayed image.
      const src = state.result ? state.result.image : { width: img.naturalWidth, height: img.naturalHeight };
      const tx = (x) => ox + (x * dw) / src.width;
      const ty = (y) => oy + (y * dh) / src.height;
      // Masks (strongest on top), then the expert outline, then boxes and labels so labels stay readable.
      // Labels are placed strongest first, so the most confident one keeps its spot above its box.
      const dets = visibleDetections();
      for (const d of [...dets].reverse()) drawMask(ctx, d, tx, ty);
      if (state.expert && state.wantExpert) {
        tracePolygons(ctx, state.expert, tx, ty);
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#fff";
        ctx.stroke();
        ctx.setLineDash([]);
      }
      for (const d of dets) drawBox(ctx, d, tx, ty);
      const placed = [];
      for (const d of dets) drawLabel(ctx, d, tx, ty, placed, h);
    }
    updatePanel();
  }

  // ---------------------------------------------------------------------------------------------- side panel
  function updateThresholdOut() {
    $("threshold-out").textContent = threshold().toFixed(2);
  }

  function updatePanel() {
    const expert = $("show-expert");
    expert.disabled = !state.expert;
    expert.checked = Boolean(state.expert) && state.wantExpert;
    $("expert-note").textContent = state.expert
      ? "Dashed white line: the radiologist-reviewed BRISC mask."
      : "Expert masks exist for the sample slices only.";

    const result = state.result;
    const list = $("detections");
    const banner = $("banner");
    list.replaceChildren();
    if (!result) {
      $("label").textContent = state.busy ? "Analysing…" : "—";
      $("score").textContent = "";
      $("bar-fill").style.width = "0";
      $("timing").textContent = "";
      banner.hidden = true;
      return;
    }

    const t = threshold();
    const dets = visibleDetections();
    const top = dets[0];
    const label = top ? nameOf(top.class_name) : NAMES.no_tumor;
    $("label").textContent = label;
    $("score").textContent = top
      ? `Confidence ${top.confidence.toFixed(2)} at threshold ${t.toFixed(2)}`
      : `No detection at or above ${t.toFixed(2)}`;
    const fill = $("bar-fill");
    fill.style.width = top ? `${(top.confidence * 100).toFixed(0)}%` : "0";
    fill.style.background = top ? colorOf(top.class_name) : "transparent";

    banner.hidden = false;
    banner.textContent = top ? `${label} · ${top.confidence.toFixed(2)}` : label;
    banner.style.color = top ? colorOf(top.class_name) : "#d6d5d0";

    if (!dets.length) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "None at this threshold";
      list.append(li);
    }
    for (const d of dets) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "name";
      const swatch = document.createElement("span");
      swatch.className = "sw";
      swatch.style.background = colorOf(d.class_name);
      name.append(swatch, nameOf(d.class_name));
      const value = document.createElement("span");
      value.className = "val mono";
      value.textContent = `${d.confidence.toFixed(2)} · ${(d.area_fraction * 100).toFixed(1)}% of slice`;
      li.append(name, value);
      list.append(li);
    }

    const ms = result.timings_ms || {};
    $("timing").textContent =
      `Inference ${Math.round(ms.inference_ms || 0)} ms, total ${Math.round(ms.total_ms || 0)} ms ` +
      `on ${providerName(result.model.provider)}`;
  }

  // ---------------------------------------------------------------------------------------------- input
  function takeFile(file) {
    if (!file) return;
    setPressed(null);
    analyse(file);
  }

  function wireDropTarget(el) {
    for (const ev of ["dragenter", "dragover"]) {
      el.addEventListener(ev, (e) => {
        e.preventDefault();
        el.classList.add("over");
      });
    }
    for (const ev of ["dragleave", "drop"]) el.addEventListener(ev, () => el.classList.remove("over"));
    el.addEventListener("drop", (e) => {
      e.preventDefault();
      takeFile(e.dataTransfer.files[0]);
    });
  }

  function init() {
    // Dropping a file anywhere else must not navigate the browser away from the page.
    for (const ev of ["dragover", "drop"]) window.addEventListener(ev, (e) => e.preventDefault());
    wireDropTarget($("drop"));
    wireDropTarget($("viewer"));
    $("drop").addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        $("file").click();
      }
    });
    $("file").addEventListener("change", (e) => {
      takeFile(e.target.files[0]);
      e.target.value = ""; // allow picking the same file again
    });
    document.addEventListener("paste", (e) => {
      const file = [...(e.clipboardData ? e.clipboardData.files : [])].find((f) => f.type.startsWith("image/"));
      if (file) takeFile(file);
    });
    $("threshold").addEventListener("input", () => {
      updateThresholdOut();
      render();
    });
    for (const id of ["show-mask", "show-outline"]) $(id).addEventListener("change", render);
    $("show-expert").addEventListener("change", (e) => {
      state.wantExpert = e.target.checked;
      render();
    });
    $("key-form").addEventListener("submit", (e) => {
      e.preventDefault();
      state.apiKey = $("api-key").value.trim() || null;
      $("key-form").hidden = true;
      if (state.retry) state.retry();
    });
    new ResizeObserver(() => render()).observe($("viewer"));

    render();
    // The model info sets the slider's range, so link parameters are applied once both have loaded.
    Promise.all([loadModel(), loadSamples()]).then(([, samples]) => applyLinkParams(samples));
  }

  init();
})();
