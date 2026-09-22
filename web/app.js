/* Ultra Fast Image Gen — canvas-first frontend. No frameworks, no build step. */

"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  stage: $("stage"), empty: $("empty"), shot: $("shot"), shotImg: $("shot-img"),
  shotSeed: $("shot-seed"), copySeed: $("copy-seed"), reuseSeed: $("reuse-seed"), download: $("download-link"),
  busy: $("busy"), busyStage: $("busy-stage"), busyDetail: $("busy-detail"),
  rail: $("rail"), error: $("error"), progress: $("progress-bar"),
  chipModel: $("chip-model"), chipSize: $("chip-size"), chipTune: $("chip-tune"),
  chipBatch: $("chip-batch"), chipMore: $("chip-more"),
  popModel: $("pop-model"), popSize: $("pop-size"), popTune: $("pop-tune"),
  popBatch: $("pop-batch"), popMore: $("pop-more"), modelList: $("model-list"),
  presets: $("presets"), width: $("width"), height: $("height"), swap: $("swap-dims"),
  steps: $("steps"), stepsOut: $("steps-out"), guidance: $("guidance"), guidanceOut: $("guidance-out"),
  seed: $("seed"), count: $("count"), countOut: $("count-out"),
  animaBlock: $("anima-block"), animaPresets: $("anima-presets"),
  loraBlock: $("lora-block"), loraPath: $("lora-path"),
  loraStrength: $("lora-strength"), loraStrengthOut: $("lora-strength-out"),
  autoSave: $("auto-save"), outputDir: $("output-dir"), openFolder: $("open-folder"),
  hfToken: $("hf-token"), saveToken: $("save-token"), tokenStatus: $("token-status"),
  addRefs: $("add-refs"), fileInput: $("file-input"), refTray: $("ref-tray"),
  prompt: $("prompt"), generate: $("generate"),
  deviceChip: $("device-chip"), storageToggle: $("storage-toggle"), storageDrawer: $("storage-drawer"),
  storageClose: $("storage-close"), storageList: $("storage-list"),
  storageTotal: $("storage-total"), storageMsg: $("storage-msg"),
  promptsToggle: $("prompts-toggle"), promptsDrawer: $("prompts-drawer"),
  promptsClose: $("prompts-close"), promptsList: $("prompts-list"),
  loraToggle: $("lora-toggle"), loraStudio: $("lora-studio"), loraClose: $("lora-close"),
  loraViewList: $("lora-view-list"), loraDatasetList: $("lora-dataset-list"),
  loraNewDataset: $("lora-new-dataset"), loraFileInput: $("lora-file-input"),
  loraViewCaption: $("lora-view-caption"), loraCaptionGrid: $("lora-caption-grid"),
  loraRecaption: $("lora-recaption"), loraSaveCaptions: $("lora-save-captions"),
  loraToTrain: $("lora-to-train"),
  loraViewTrain: $("lora-view-train"), loraTrigger: $("lora-trigger"),
  loraSteps: $("lora-steps"), loraStepsOut: $("lora-steps-out"), loraEtaHint: $("lora-eta-hint"),
  loraStartTrain: $("lora-start-train"),
  loraViewProgress: $("lora-view-progress"), loraProgressStage: $("lora-progress-stage"),
  loraProgressDetail: $("lora-progress-detail"), loraProgressBar: $("lora-progress-bar"),
  loraViewDone: $("lora-view-done"), loraUse: $("lora-use"), loraError: $("lora-error"),
};

let MODELS = [];
let modelId = null;
let refImages = [];
let lastSeed = null;
let pollTimer = null;
let pollFails = 0;
let activeJob = null;
let animaPreset = "Balanced";

const SETTINGS_KEY = "ufig-settings-v2";
const SAVED_PROMPTS_KEY = "ufig-saved-prompts-v1";
const SAVED_PROMPTS_MAX = 200;
const POPS = [
  ["chipModel", "popModel"], ["chipSize", "popSize"], ["chipTune", "popTune"],
  ["chipBatch", "popBatch"], ["chipMore", "popMore"],
];

/* ---------------- helpers ---------------- */

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const model = () => MODELS.find((m) => m.id === modelId);

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = !msg;
}

/* ---------------- settings ---------------- */

function saveSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({
    model: modelId, prompt: els.prompt.value,
    width: els.width.value, height: els.height.value,
    steps: els.steps.value, guidance: els.guidance.value, count: els.count.value,
    autoSave: els.autoSave.checked, outputDir: els.outputDir.value,
    loraPath: els.loraPath.value, loraStrength: els.loraStrength.value, animaPreset,
  }));
}

function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || null; }
  catch { return null; }
}

/* ---------------- saved prompts ---------------- */

function loadSavedPrompts() {
  try { return JSON.parse(localStorage.getItem(SAVED_PROMPTS_KEY)) || []; }
  catch { return []; }
}

function writeSavedPrompts(list) {
  localStorage.setItem(SAVED_PROMPTS_KEY, JSON.stringify(list.slice(0, SAVED_PROMPTS_MAX)));
}

function saveCurrentPrompt() {
  const text = els.prompt.value.trim();
  if (!text) return;
  const list = loadSavedPrompts().filter((p) => p.text !== text);
  list.unshift({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, text, savedAt: Date.now() });
  writeSavedPrompts(list);
  renderSavedPrompts();
}

function deleteSavedPrompt(id) {
  writeSavedPrompts(loadSavedPrompts().filter((p) => p.id !== id));
  renderSavedPrompts();
}

function renderSavedPrompts() {
  const list = loadSavedPrompts();
  els.promptsList.textContent = "";
  for (const p of list) {
    const li = document.createElement("li");
    li.className = "prompt-item";
    const text = document.createElement("span");
    text.className = "p-text";
    text.title = p.text;
    text.textContent = p.text;
    li.appendChild(text);
    const del = document.createElement("button");
    del.className = "icon-link";
    del.type = "button";
    del.textContent = "Delete";
    let armed = false;
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!armed) {
        armed = true;
        del.textContent = "Confirm?";
        del.classList.add("danger");
        setTimeout(() => { armed = false; del.textContent = "Delete"; del.classList.remove("danger"); }, 3000);
        return;
      }
      deleteSavedPrompt(p.id);
    });
    li.appendChild(del);
    li.addEventListener("click", () => {
      els.prompt.value = p.text;
      autoGrow();
      saveSettings();
      els.promptsDrawer.hidden = true;
      startGeneration();
    });
    els.promptsList.appendChild(li);
  }
}

/* ---------------- popovers ---------------- */

function closePops(except) {
  for (const [chip, pop] of POPS) {
    if (pop === except) continue;
    els[pop].hidden = true;
    els[chip].setAttribute("aria-expanded", "false");
  }
}

function bindPop(chipKey, popKey) {
  els[chipKey].addEventListener("click", () => {
    const open = !els[popKey].hidden;
    closePops();
    els[popKey].hidden = open;
    els[chipKey].setAttribute("aria-expanded", String(!open));
  });
}

/* ---------------- chips / state sync ---------------- */

function syncChips() {
  const m = model();
  if (!m) return;
  els.chipModel.textContent = "";
  if (m.setup_required) {
    const w = document.createElement("span");
    w.className = "warn";
    w.textContent = "⚠ ";
    els.chipModel.appendChild(w);
  }
  els.chipModel.appendChild(document.createTextNode(m.label));
  els.chipSize.textContent = `${els.width.value}×${els.height.value}`;
  const g = Number(els.guidance.value);
  els.chipTune.textContent = `${els.steps.value} steps${g ? ` · cfg ${g}` : ""}${els.seed.value ? ` · seed ${els.seed.value}` : ""}`;
  els.chipBatch.textContent = `×${els.count.value}`;

  els.stepsOut.textContent = els.steps.value;
  els.guidanceOut.textContent = els.guidance.value;
  els.countOut.textContent = els.count.value;
  els.loraStrengthOut.textContent = Number(els.loraStrength.value).toFixed(2);

  for (const b of els.presets.querySelectorAll(".preset")) {
    const on = b.dataset.size === els.width.value && b.dataset.size === els.height.value;
    b.setAttribute("aria-pressed", String(on));
  }
  els.addRefs.hidden = !m.img2img;
  if (!m.img2img && refImages.length) { refImages = []; renderRefs(); }
  els.animaBlock.hidden = m.id !== "anima";
  els.loraBlock.hidden = !m.lora;
}

function applyModelDefaults(m) {
  els.width.value = m.defaults.width;
  els.height.value = m.defaults.height;
  els.steps.value = m.defaults.steps;
  els.guidance.value = m.defaults.guidance;
}

function fmtBytes(n) {
  if (n == null) return "";
  return n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : `${Math.round(n / 1024 ** 2)} MB`;
}

function renderModelControls(m, ctl) {
  ctl.textContent = "";
  if (!m.managed) {
    ctl.appendChild(document.createTextNode("downloads on first use"));
    return;
  }
  const dl = DOWNLOADS[m.id];
  if (dl?.status === "running") {
    const span = document.createElement("span");
    span.className = "dl-progress";
    span.dataset.dl = m.id;
    span.textContent = dlText(dl);
    ctl.appendChild(span);
    return;
  }
  if (dl?.status === "error") {
    const span = document.createElement("span");
    span.className = "dl-error";
    span.textContent = dl.error || "download failed";
    ctl.appendChild(span);
  }
  if (m.downloaded) {
    ctl.appendChild(document.createTextNode(m.size_str || "downloaded"));
    const del = document.createElement("button");
    del.className = "icon-link";
    del.type = "button";
    del.textContent = "Delete";
    let armed = false;
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!armed) {          // inline confirm: first click arms, second deletes
        armed = true;
        del.textContent = "Confirm?";
        del.classList.add("danger");
        setTimeout(() => { armed = false; del.textContent = "Delete"; del.classList.remove("danger"); }, 3000);
        return;
      }
      del.disabled = true;
      try {
        await api(`/api/models/${m.id}/delete`, { method: "POST" });
        await refreshModels();
      } catch (err) {
        del.disabled = false;
        ctl.appendChild(document.createTextNode(` ${err.message}`));
      }
    });
    ctl.appendChild(del);
  } else {
    const get = document.createElement("button");
    get.className = "icon-link";
    get.type = "button";
    get.textContent = dl?.status === "error" ? "Retry" : "Download";
    get.addEventListener("click", async (e) => {
      e.stopPropagation();
      get.disabled = true;
      try {
        await api(`/api/models/${m.id}/download`, { method: "POST" });
        DOWNLOADS[m.id] = { status: "running", done: 0, total: null };
        renderModelControls(m, ctl);
      } catch (err) {
        get.disabled = false;
        ctl.appendChild(document.createTextNode(` ${err.message}`));
      }
    });
    ctl.appendChild(get);
  }
}

function renderModelList() {
  els.modelList.textContent = "";
  for (const m of MODELS) {
    const li = document.createElement("li");
    li.classList.toggle("selected", m.id === modelId);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "m-select";
    b.setAttribute("aria-pressed", String(m.id === modelId));

    const head = document.createElement("div");
    head.className = "m-head";
    const name = document.createElement("span");
    name.className = "m-name";
    name.textContent = m.label;
    head.appendChild(name);
    if (m.tag) {
      const tag = document.createElement("span");
      tag.className = "m-tag";
      tag.textContent = m.tag;
      head.appendChild(tag);
    }

    const note = document.createElement("span");
    note.className = "m-note";
    if (m.setup_required) {
      const warn = document.createElement("span");
      warn.className = "warn";
      warn.textContent = "Setup required — run scripts/setup_mflux_hs.sh. ";
      note.appendChild(warn);
    }
    note.appendChild(document.createTextNode(m.note || ""));
    note.title = m.note || "";

    b.append(head, note);
    b.addEventListener("click", () => {
      modelId = m.id;
      applyModelDefaults(m);
      renderModelList();
      syncChips();
      saveSettings();
      closePops();
    });

    const ctl = document.createElement("div");
    ctl.className = "m-ctl";
    renderModelControls(m, ctl);

    li.append(b, ctl);
    els.modelList.appendChild(li);
  }
}

/* ---------------- download polling ---------------- */

let DOWNLOADS = {};

function dlText(dl) {
  if (dl.total) {
    const pct = Math.min(99, Math.round((dl.done / dl.total) * 100));
    return `downloading ${pct}% · ${fmtBytes(dl.done)} / ${fmtBytes(dl.total)}`;
  }
  return `downloading · ${fmtBytes(dl.done)}`;
}

async function refreshModels() {
  MODELS = await api("/api/models");
  renderModelList();
  syncChips();
}

async function pollDownloads() {
  const anyRunning = Object.values(DOWNLOADS).some((d) => d.status === "running");
  if (els.popModel.hidden && !anyRunning) return;
  let next;
  try { next = await api("/api/downloads"); }
  catch { return; }
  const finished = Object.keys(next).some(
    (id) => DOWNLOADS[id]?.status === "running" && next[id].status !== "running"
  );
  DOWNLOADS = next;
  if (finished) {
    await refreshModels().catch(() => {});
    return;
  }
  // update progress text in place to keep scroll position
  for (const [id, dl] of Object.entries(DOWNLOADS)) {
    if (dl.status !== "running") continue;
    const span = els.modelList.querySelector(`[data-dl="${id}"]`);
    if (span) span.textContent = dlText(dl);
  }
}

function renderAnimaPresets() {
  const m = MODELS.find((x) => x.id === "anima");
  els.animaPresets.textContent = "";
  for (const name of m?.anima_presets || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset";
    b.textContent = name;
    b.setAttribute("aria-pressed", String(name === animaPreset));
    b.addEventListener("click", () => { animaPreset = name; renderAnimaPresets(); saveSettings(); });
    els.animaPresets.appendChild(b);
  }
}

/* ---------------- reference images ---------------- */

function renderRefs() {
  els.refTray.textContent = "";
  els.refTray.hidden = refImages.length === 0;
  refImages.forEach((src, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = src;
    img.alt = `Reference ${i + 1}`;
    const rm = document.createElement("button");
    rm.className = "remove";
    rm.type = "button";
    rm.textContent = "×";
    rm.setAttribute("aria-label", `Remove reference ${i + 1}`);
    rm.addEventListener("click", () => { refImages.splice(i, 1); renderRefs(); });
    wrap.append(img, rm);
    els.refTray.appendChild(wrap);
  });
}

function addFiles(files) {
  if (!model()?.img2img) return;
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    if (refImages.length >= 6) break;
    const reader = new FileReader();
    reader.onload = () => { refImages.push(reader.result); renderRefs(); };
    reader.readAsDataURL(f);
  }
}

/* ---------------- generation ---------------- */

function setBusy(busy) {
  els.generate.disabled = busy;
  els.busy.hidden = !busy;
  if (!busy) {
    els.progress.classList.remove("indeterminate");
    els.progress.style.width = "0%";
  }
}

async function startGeneration() {
  // generate is disabled from setBusy(true) until the job reaches a terminal
  // state, so this also guards the Enter-during-POST race
  if (activeJob || els.generate.disabled) return;
  showError("");
  const m = model();
  if (!m) return;
  if (!els.prompt.value.trim()) { showError("Write a prompt first."); els.prompt.focus(); return; }

  const body = {
    model: m.id,
    prompt: els.prompt.value.trim(),
    width: Number(els.width.value),
    height: Number(els.height.value),
    steps: Number(els.steps.value),
    guidance: Number(els.guidance.value),
    seed: els.seed.value === "" ? null : Number(els.seed.value),
    count: Number(els.count.value),
    auto_save: els.autoSave.checked,
    output_dir: els.outputDir.value || null,
  };
  if (m.img2img && refImages.length) body.input_images = refImages;
  if (m.lora && els.loraPath.value) {
    body.lora_path = els.loraPath.value;
    body.lora_strength = Number(els.loraStrength.value);
  }
  if (m.id === "anima") body.anima_preset = animaPreset;

  closePops();
  setBusy(true);
  els.busyStage.textContent = "Queued";
  els.busyDetail.textContent = "";
  try {
    const { job_id } = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    activeJob = job_id;
    pollFails = 0;
    clearInterval(pollTimer);
    pollTimer = setInterval(() => pollJob(job_id), 600);
  } catch (e) {
    setBusy(false);
    showError(e.message);
  }
}

function abandonJob(message) {
  clearInterval(pollTimer);
  activeJob = null;
  setBusy(false);
  showError(message);
}

async function pollJob(jobId) {
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
    pollFails = 0;
  } catch (e) {
    if (jobId !== activeJob) return;
    // A 404 means the server restarted and forgot the job; transient network
    // errors get a grace window (~12s) before giving up.
    if (e.status === 404) abandonJob("The server restarted and lost this job — try again.");
    else if (++pollFails > 20) abandonJob("Lost contact with the server — is it still running?");
    return;
  }
  if (jobId !== activeJob) return;

  els.busyStage.textContent = job.queue_position > 0
    ? `Queued (#${job.queue_position + 1})`
    : job.stage;
  els.busyDetail.textContent = job.elapsed != null ? `${job.elapsed.toFixed(0)}s` : "";

  if (job.progress == null && job.status === "running") {
    els.progress.style.width = ""; // leftover inline width would hide the slide animation
    els.progress.classList.add("indeterminate");
  } else if (job.progress != null) {
    els.progress.classList.remove("indeterminate");
    els.progress.style.width = `${Math.round(job.progress * 100)}%`;
  }

  if (job.images.length) addToRail(job.images);

  if (["done", "error", "cancelled"].includes(job.status)) {
    clearInterval(pollTimer);
    activeJob = null;
    setBusy(false);
    if (job.status === "error") showError(job.error || "Generation failed.");
    if (job.images.length) {
      lastSeed = job.images[job.images.length - 1].seed;
      selectImage(job.images[job.images.length - 1]);
    }
    refreshModels().catch(() => {}); // a first-use download may have changed flags
  }
}

/* ---------------- LoRA Studio ---------------- */

// Measured on this session's hardware (Z-Image-Turbo, bf16, rank 16, res 512,
// batch 1 + grad-accum 4) — a rough ETA hint only, not a guarantee.
const LORA_SEC_PER_STEP = 83;

let loraDatasetId = null;
let loraImages = [];
let loraCaptions = {};
let loraPollTimer = null;
let loraTrainedPath = null;

function loraShowView(name) {
  for (const v of document.querySelectorAll(".lora-view")) v.hidden = true;
  els[`loraView${name[0].toUpperCase()}${name.slice(1)}`].hidden = false;
}

function loraShowError(msg) {
  els.loraError.textContent = msg;
  els.loraError.hidden = !msg;
}

async function loraOpen() {
  els.loraStudio.hidden = false;
  loraShowError("");
  loraShowView("list");
  await loraRefreshDatasetList().catch((e) => loraShowError(e.message));
}

function loraClose() {
  els.loraStudio.hidden = true;
  clearInterval(loraPollTimer);
}

async function loraRefreshDatasetList() {
  const datasets = await api("/api/lora/datasets");
  els.loraDatasetList.textContent = "";
  for (const d of datasets) {
    const li = document.createElement("li");
    li.className = "dataset-item";
    const name = document.createElement("span");
    name.className = "s-name";
    name.textContent = `${d.num_images} image${d.num_images === 1 ? "" : "s"}`;
    const badge = document.createElement("span");
    badge.className = "dataset-badge";
    badge.textContent = d.trained ? "trained" : d.captioned ? "captioned" : "new";
    const del = document.createElement("button");
    del.className = "icon-link";
    del.type = "button";
    del.textContent = "Delete";
    let armed = false;
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!armed) {
        armed = true;
        del.textContent = "Confirm?";
        del.classList.add("danger");
        setTimeout(() => { armed = false; del.textContent = "Delete"; del.classList.remove("danger"); }, 3000);
        return;
      }
      await api(`/api/lora/datasets/${d.id}`, { method: "DELETE" });
      loraRefreshDatasetList();
    });
    li.append(name, badge, del);
    li.addEventListener("click", () => loraOpenDataset(d));
    els.loraDatasetList.appendChild(li);
  }
}

async function loraOpenDataset(d) {
  loraDatasetId = d.id;
  loraImages = Array.from({ length: d.num_images }, (_, i) => null); // filled below
  const imgsResp = await api(`/api/lora/datasets/${d.id}/captions`);
  loraCaptions = imgsResp.captions || {};
  // The API doesn't have a dedicated "list images" endpoint beyond upload's
  // response, so recover the filenames from the captions keys when present;
  // otherwise re-caption is the only way in (acceptable — it's the next step anyway).
  loraImages = Object.keys(loraCaptions);
  if (loraImages.length) {
    loraRenderCaptionGrid();
    loraShowView("caption");
  } else {
    loraStartCaptioning();
  }
}

async function loraCreateDataset(files) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  loraShowError("");
  try {
    const r = await api("/api/lora/datasets", { method: "POST", body: form });
    loraDatasetId = r.dataset_id;
    loraImages = r.images;
    loraCaptions = {};
    loraStartCaptioning();
  } catch (e) {
    loraShowError(e.message);
  }
}

function loraPollJob(jobId, { onTick, onDone }) {
  clearInterval(loraPollTimer);
  loraPollTimer = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (e) {
      clearInterval(loraPollTimer);
      loraShowError(e.message);
      return;
    }
    onTick(job);
    if (["done", "error", "cancelled"].includes(job.status)) {
      clearInterval(loraPollTimer);
      if (job.status === "error") loraShowError(job.error || "Job failed.");
      else onDone(job);
    }
  }, 1500);
}

async function loraStartCaptioning() {
  loraShowView("progress");
  els.loraProgressStage.textContent = "Captioning";
  els.loraProgressDetail.textContent = "";
  els.loraProgressBar.style.width = "0%";
  let jobId;
  try {
    ({ job_id: jobId } = await api(`/api/lora/datasets/${loraDatasetId}/caption`, { method: "POST" }));
  } catch (e) {
    loraShowError(e.message);
    return;
  }
  loraPollJob(jobId, {
    onTick: (job) => {
      els.loraProgressStage.textContent = job.stage || "Captioning";
      els.loraProgressBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    },
    onDone: (job) => {
      loraCaptions = job.captions || {};
      loraImages = Object.keys(loraCaptions);
      loraRenderCaptionGrid();
      loraShowView("caption");
    },
  });
}

function loraRenderCaptionGrid() {
  els.loraCaptionGrid.textContent = "";
  for (const name of loraImages) {
    const card = document.createElement("div");
    card.className = "lora-caption-card";
    const img = document.createElement("img");
    img.src = `/api/lora/datasets/${loraDatasetId}/images/${name}`;
    img.alt = name;
    const ta = document.createElement("textarea");
    ta.value = loraCaptions[name] || "";
    ta.addEventListener("input", () => { loraCaptions[name] = ta.value; });
    card.append(img, ta);
    els.loraCaptionGrid.appendChild(card);
  }
}

async function loraSaveCaptions() {
  await api(`/api/lora/datasets/${loraDatasetId}/captions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ captions: loraCaptions }),
  });
}

function loraUpdateEtaHint() {
  const steps = Number(els.loraSteps.value);
  els.loraStepsOut.textContent = steps;
  const secs = steps * LORA_SEC_PER_STEP;
  const mins = Math.round(secs / 60);
  const hrs = (secs / 3600).toFixed(1);
  els.loraEtaHint.textContent = secs < 3600
    ? `~${mins} min at the rate measured earlier this session — actual time varies.`
    : `~${hrs} h at the rate measured earlier this session — actual time varies.`;
}

async function loraStartTraining() {
  const trigger = els.loraTrigger.value.trim();
  if (!trigger) { loraShowError("Trigger token is required."); return; }
  loraShowError("");
  try {
    await loraSaveCaptions();
  } catch (e) {
    loraShowError(e.message);
    return;
  }
  loraShowView("progress");
  els.loraProgressStage.textContent = "Starting";
  els.loraProgressDetail.textContent = "";
  els.loraProgressBar.style.width = "0%";
  let jobId;
  try {
    ({ job_id: jobId } = await api(`/api/lora/datasets/${loraDatasetId}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trigger, max_steps: Number(els.loraSteps.value) }),
    }));
  } catch (e) {
    loraShowError(e.message);
    return;
  }
  loraPollJob(jobId, {
    onTick: (job) => {
      els.loraProgressStage.textContent = job.stage || "Training";
      const bits = [];
      if (job.step != null && job.total_steps) bits.push(`step ${job.step}/${job.total_steps}`);
      if (job.loss != null) bits.push(`loss ${job.loss.toFixed(3)}`);
      if (job.eta) bits.push(`eta ${job.eta}`);
      els.loraProgressDetail.textContent = bits.join(" · ");
      els.loraProgressBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    },
    onDone: (job) => {
      loraTrainedPath = job.lora_path;
      loraShowView("done");
    },
  });
}

function loraUseInStudio() {
  const m = MODELS.find((x) => x.id === "zimage-full");
  if (m) {
    modelId = m.id;
    applyModelDefaults(m);
    renderModelList();
    syncChips();
  }
  els.loraPath.value = loraTrainedPath || "";
  els.loraStrength.value = 1;
  saveSettings();
  loraClose();
  els.prompt.focus();
}

/* ---------------- rail / viewer ---------------- */

const seenUrls = new Set();

function addToRail(images) {
  for (const img of images) {
    if (seenUrls.has(img.url)) continue;
    seenUrls.add(img.url);
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `View image with seed ${img.seed}`);
    const t = document.createElement("img");
    t.src = img.url;
    t.alt = "";
    b.appendChild(t);
    b.addEventListener("click", () => selectImage(img, b));
    els.rail.prepend(b);
    if (els.rail.children.length > 80) els.rail.lastChild.remove();
  }
}

function selectImage(img, btn) {
  els.empty.hidden = true;
  els.shot.hidden = false;
  els.shotImg.src = img.url;
  els.shotSeed.textContent = `seed ${img.seed}${img.saved ? " · saved" : ""}`;
  els.shotSeed.dataset.seed = img.seed;
  els.shotSeed.title = img.saved || "";
  els.download.href = img.url;
  els.download.setAttribute("download", `ufig-${img.seed}.png`);
  for (const child of els.rail.children) child.classList.remove("active");
  if (btn) btn.classList.add("active");
  else {
    const match = [...els.rail.children].find((c) => c.querySelector("img")?.src.endsWith(img.url));
    match?.classList.add("active");
  }
}

/* ---------------- storage drawer ---------------- */

async function refreshStorage() {
  const data = await api("/api/storage");
  els.storageTotal.textContent = data.models.length
    ? `Total: ${data.total_str}`
    : "Nothing downloaded yet — models download on first use.";
  els.storageList.textContent = "";
  for (const m of data.models) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "s-name";
    name.title = m.repo_id;
    name.textContent = m.name;
    const size = document.createElement("span");
    size.className = "s-size tab-nums";
    size.textContent = m.size_str;
    const del = document.createElement("button");
    del.className = "icon-link";
    del.type = "button";
    del.textContent = "Delete";
    let armed = false;
    del.addEventListener("click", async () => {
      if (!armed) {            // inline confirm: first click arms, second deletes
        armed = true;
        del.textContent = "Confirm?";
        del.classList.add("danger");
        setTimeout(() => { armed = false; del.textContent = "Delete"; del.classList.remove("danger"); }, 3000);
        return;
      }
      del.disabled = true;
      try {
        const r = await api("/api/storage/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key: m.key }),
        });
        els.storageMsg.textContent = r.message;
      } catch (e) {
        els.storageMsg.textContent = e.message;
      }
      refreshStorage();
    });
    li.append(name, size, del);
    els.storageList.appendChild(li);
  }
}

/* ---------------- init ---------------- */

function autoGrow() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 130)}px`;
}

async function init() {
  const [status, models] = await Promise.all([api("/api/status"), api("/api/models")]);
  MODELS = models;
  els.deviceChip.textContent = status.devices[0] || "cpu";
  els.outputDir.value = status.default_output_dir;

  const saved = loadSettings();
  modelId = saved && MODELS.some((m) => m.id === saved.model) ? saved.model : MODELS[0].id;
  applyModelDefaults(model());

  if (saved) {
    els.prompt.value = saved.prompt || "";
    if (saved.width) els.width.value = saved.width;
    if (saved.height) els.height.value = saved.height;
    if (saved.steps) els.steps.value = saved.steps;
    if (saved.guidance != null) els.guidance.value = saved.guidance;
    if (saved.count) els.count.value = saved.count;
    els.autoSave.checked = !!saved.autoSave;
    if (saved.outputDir) els.outputDir.value = saved.outputDir;
    if (saved.loraPath) els.loraPath.value = saved.loraPath;
    if (saved.loraStrength) els.loraStrength.value = saved.loraStrength;
    if (saved.animaPreset) animaPreset = saved.animaPreset;
  }

  renderModelList();
  renderAnimaPresets();
  syncChips();
  autoGrow();

  for (const [chip, pop] of POPS) bindPop(chip, pop);
  setInterval(pollDownloads, 2000);
  els.chipModel.addEventListener("click", () => {
    if (!els.popModel.hidden) pollDownloads();
  });
  document.addEventListener("click", (e) => {
    if (!e.target.isConnected) return; // re-rendered controls (e.g. presets) detach mid-bubble
    if (!e.target.closest(".pop") && !e.target.closest(".chip")) closePops();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closePops();
      els.storageDrawer.hidden = true;
      els.promptsDrawer.hidden = true;
      if (!els.loraStudio.hidden) loraClose();
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      saveCurrentPrompt();
      els.promptsDrawer.hidden = false;
    }
  });

  for (const el of [els.steps, els.guidance, els.count, els.loraStrength, els.width, els.height, els.seed]) {
    el.addEventListener("input", () => { syncChips(); saveSettings(); });
  }
  for (const el of [els.outputDir, els.loraPath]) el.addEventListener("change", saveSettings);
  els.autoSave.addEventListener("change", saveSettings);
  els.prompt.addEventListener("input", () => { autoGrow(); });
  els.prompt.addEventListener("change", saveSettings);

  els.presets.addEventListener("click", (e) => {
    const b = e.target.closest(".preset");
    if (!b) return;
    els.width.value = b.dataset.size;
    els.height.value = b.dataset.size;
    syncChips();
    saveSettings();
  });

  els.swap.addEventListener("click", () => {
    [els.width.value, els.height.value] = [els.height.value, els.width.value];
    syncChips();
    saveSettings();
  });

  els.reuseSeed.addEventListener("click", () => {
    const s = els.shotSeed.dataset.seed;
    if (s) { els.seed.value = s; syncChips(); saveSettings(); }
  });

  els.copySeed.addEventListener("click", () => {
    navigator.clipboard?.writeText(els.shotSeed.dataset.seed || "");
  });

  els.generate.addEventListener("click", startGeneration);
  els.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startGeneration(); }
  });

  for (const s of document.querySelectorAll(".sample")) {
    s.addEventListener("click", () => {
      els.prompt.value = s.textContent;
      autoGrow();
      els.prompt.focus();
      saveSettings();
    });
  }

  // reference images: button, file picker, drag-drop onto the page
  els.addRefs.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => { addFiles(els.fileInput.files); els.fileInput.value = ""; });
  document.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (model()?.img2img) els.addRefs.classList.add("dragover");
  });
  document.addEventListener("dragleave", (e) => {
    if (e.relatedTarget === null) els.addRefs.classList.remove("dragover");
  });
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    els.addRefs.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
  });

  els.openFolder.addEventListener("click", () =>
    api("/api/open_folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: els.outputDir.value || null }),
    }).catch((e) => showError(e.message))
  );

  // Hugging Face token
  els.tokenStatus.textContent = status.hf_token_set
    ? "A token is saved (stored in .env)."
    : "No token set yet.";
  els.saveToken.addEventListener("click", async () => {
    els.saveToken.disabled = true;
    els.tokenStatus.textContent = "Checking token…";
    try {
      const r = await api("/api/settings/hf_token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: els.hfToken.value }),
      });
      els.hfToken.value = "";
      els.tokenStatus.textContent = r.user
        ? `Saved — signed in as ${r.user}.`
        : "Saved (stored in .env).";
    } catch (e) {
      els.tokenStatus.textContent = e.message;
    }
    els.saveToken.disabled = false;
  });

  els.storageToggle.addEventListener("click", () => {
    els.storageDrawer.hidden = !els.storageDrawer.hidden;
    if (!els.storageDrawer.hidden) refreshStorage().catch((e) => { els.storageMsg.textContent = e.message; });
  });
  els.storageClose.addEventListener("click", () => { els.storageDrawer.hidden = true; });

  els.promptsToggle.addEventListener("click", () => {
    els.promptsDrawer.hidden = !els.promptsDrawer.hidden;
    if (!els.promptsDrawer.hidden) renderSavedPrompts();
  });
  els.promptsClose.addEventListener("click", () => { els.promptsDrawer.hidden = true; });

  els.loraToggle.addEventListener("click", () => loraOpen());
  els.loraClose.addEventListener("click", () => loraClose());
  els.loraNewDataset.addEventListener("click", () => els.loraFileInput.click());
  els.loraFileInput.addEventListener("change", () => {
    if (els.loraFileInput.files.length) loraCreateDataset(els.loraFileInput.files);
    els.loraFileInput.value = "";
  });
  els.loraRecaption.addEventListener("click", () => loraStartCaptioning());
  els.loraSaveCaptions.addEventListener("click", () =>
    loraSaveCaptions().catch((e) => loraShowError(e.message))
  );
  els.loraToTrain.addEventListener("click", async () => {
    try {
      await loraSaveCaptions();
    } catch (e) {
      loraShowError(e.message);
      return;
    }
    loraShowError("");
    loraUpdateEtaHint();
    loraShowView("train");
  });
  els.loraSteps.addEventListener("input", loraUpdateEtaHint);
  els.loraStartTrain.addEventListener("click", () => loraStartTraining());
  els.loraUse.addEventListener("click", () => loraUseInStudio());
}

init().catch((e) => showError(`Failed to load: ${e.message}`));
