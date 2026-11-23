// Detection Alerts - single-page UI (no build step, no libraries)

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const PALETTE = ["#90ee90", "#add8e6", "#ffff99", "#ffc0cb", "#dda0dd", "#ff5000",
                 "#ffa500", "#00ced1", "#9acd32", "#ff69b4", "#87cefa", "#f0e68c"];
const SOURCE_HINTS = {
  video: "/path/to/video.mp4",
  rtsp: "rtsp://user:pass@192.168.1.10:554/stream",
  webcam: "0",
};

const VIEWS = ["dashboard", "violations", "cameras", "model"];

const state = {
  view: "dashboard",
  focus: null,            // camera id shown big on the dashboard
  focusPushed: false,     // true when we opened it ourselves (so Back returns to the grid)
  prevFilterCamera: "",
  status: null,
  cameras: [],
  alerts: [],
  filterCamera: "",
  editingCameraId: null,
  sound: readPref("sound", "1") === "1",
};

function readPref(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function writePref(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage blocked */ }
}

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
  return data;
}

function toast(message, kind = "info", ms = 4000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function fmtTime(iso) {
  const d = new Date(iso);
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay ? d.toLocaleTimeString() : d.toLocaleString();
}

// ===================== navigation =====================

// The URL hash is the page: "" = dashboard, "#violations", "#cameras", "#model",
// "#camera/<id>" = one camera big. Using the hash makes the browser Back button work.
function go(hash) {
  location.hash = hash;
}

function route() {
  const hash = decodeURIComponent(location.hash.slice(1));
  let view = hash, focus = null;
  if (hash.startsWith("camera/")) {
    view = "dashboard";
    focus = hash.slice("camera/".length);
  }
  if (!VIEWS.includes(view)) view = "dashboard";

  state.view = view;
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  setFocus(focus);
  if (view === "violations") loadViolations();
  if (view === "cameras") loadCameras();
  if (view === "model") loadModel();
}
window.addEventListener("hashchange", route);

$$(".tab").forEach((t) => t.addEventListener("click", () => go(t.dataset.view === "dashboard" ? "" : t.dataset.view)));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => go(b.dataset.goto)));
$$("[data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));

// ===================== engine + camera grid =====================

async function refreshStatus() {
  try {
    const s = await api("GET", "/api/engine/status");
    state.status = s;
    renderEngine(s);
    renderGrid(s.cameras);
    renderCounts(s.alerts);
  } catch { /* server restarting - try again next tick */ }
}

function renderEngine(s) {
  const pill = $("#engine-state");
  pill.textContent = s.running ? "Detection running" : "Detection stopped";
  pill.classList.toggle("on", s.running);
  const btn = $("#engine-toggle");
  btn.textContent = s.running ? "Stop" : "Start";
  btn.classList.toggle("primary", !s.running);
}

$("#engine-toggle").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    await api("POST", state.status?.running ? "/api/engine/stop" : "/api/engine/start");
    await refreshStatus();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    btn.disabled = false;
  }
});

function renderGrid(cameras) {
  const grid = $("#camera-grid");
  const active = cameras.filter((c) => c.status !== "disabled");
  $("#grid-empty").classList.toggle("hidden", active.length > 0);

  const ids = active.map((c) => c.id).join("|");
  if (grid.dataset.ids !== ids) {
    grid.dataset.ids = ids;
    grid.innerHTML = active.map((c) => `
      <div class="tile" data-id="${esc(c.id)}" title="Double-click to enlarge">
        <div class="tile-head">
          <span class="dot"></span><b></b><span class="loc"></span>
          <button class="icon-btn expand" data-act="expand" title="Enlarge">⤢</button>
        </div>
        <div class="tile-video"><img alt=""></div>
        <div class="tile-foot"><span class="st"></span><span class="nums"></span></div>
      </div>`).join("");
    $$(".tile", grid).forEach((t) => startTileStream($("img", t), t.dataset.id));
    setFocus(state.focus);
  }

  for (const c of active) {
    const tile = grid.querySelector(`.tile[data-id="${CSS.escape(c.id)}"]`);
    if (!tile) continue;
    $("b", tile).textContent = c.name;
    $(".loc", tile).textContent = c.location;
    const dot = $(".dot", tile);
    dot.className = "dot " + (c.status === "error" ? "bad"
      : c.status === "running" && c.connected ? "ok"
      : c.status === "stopped" ? "" : "wait");
    const st = $(".st", tile);
    st.textContent = c.error || c.status;
    st.title = c.error || "";
    st.classList.toggle("err", !!c.error);
    $(".nums", tile).textContent = `${c.fps} fps · ${c.alerts} alert${c.alerts === 1 ? "" : "s"}`;
  }

  syncCameraOptions($("#alert-filter-camera"), cameras, state.filterCamera);
  syncCameraOptions($("#v-camera"), cameras);
  if (state.focus) {
    const cam = cameras.find((c) => c.id === state.focus);
    if (cam) $("#focus-title").textContent = `${cam.name} · ${cam.location}`;
    else exitFocus();   // camera was deleted or disabled
  }
}

// Keep a camera <select> in sync with the camera list without losing its selection.
function syncCameraOptions(sel, cameras, value = sel.value) {
  const opts = cameras.map((c) => c.id + c.name).join("|");
  if (sel.dataset.opts === opts) return;
  sel.dataset.opts = opts;
  sel.innerHTML = `<option value="">All cameras</option>` +
    cameras.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("");
  sel.value = value;
}

// ---------- one camera big ----------

function setFocus(camId) {
  const changed = camId !== state.focus;
  state.focus = camId;
  if (!camId) state.focusPushed = false;
  const grid = $("#camera-grid");
  grid.classList.toggle("focused", !!camId);
  $$(".tile", grid).forEach((t) => {
    const big = t.dataset.id === camId;
    t.classList.toggle("big", big);
    t.title = camId ? "Double-click to go back" : "Double-click to enlarge";
    $(".expand", t).textContent = big ? "✕" : "⤢";
    $(".expand", t).title = big ? "Back to all cameras" : "Enlarge";
  });
  $("#focus-bar").classList.toggle("hidden", !camId);
  if (!changed) return;

  // the alert panel follows the big camera, and goes back to its old filter afterwards
  if (camId) {
    state.prevFilterCamera = state.filterCamera;
    state.filterCamera = camId;
  } else {
    state.filterCamera = state.prevFilterCamera;
  }
  $("#alert-filter-camera").value = state.filterCamera;
  loadAlerts();
}

function enterFocus(camId) {
  state.focusPushed = true;
  go(`camera/${camId}`);
}

function exitFocus() {
  if (state.focusPushed) history.back();   // same as pressing the browser Back button
  else go("");
}

$("#camera-grid").addEventListener("dblclick", (e) => {
  const tile = e.target.closest(".tile");
  if (!tile || e.target.closest("button")) return;
  window.getSelection()?.removeAllRanges();
  if (state.focus) exitFocus();
  else enterFocus(tile.dataset.id);
});

$("#camera-grid").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-act=expand]");
  if (!btn) return;
  if (state.focus) exitFocus();
  else enterFocus(btn.closest(".tile").dataset.id);
});

$("#focus-back").addEventListener("click", exitFocus);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.focus && !document.querySelector("dialog[open]")) exitFocus();
});

// Each tile pulls the latest annotated frame as a JPEG, one request at a time.
// (Polling instead of MJPEG avoids the browser's ~6 open-connections-per-host limit.)
// The big view asks for the full-resolution frame; hidden tiles pause.
function startTileStream(img, camId) {
  const next = () => {
    if (!img.isConnected) return;
    const hidden = state.focus && state.focus !== camId;
    if (document.hidden || state.view !== "dashboard" || hidden) { setTimeout(next, 500); return; }
    const full = state.focus === camId ? "&full=1" : "";
    img.src = `/api/cameras/${encodeURIComponent(camId)}/frame.jpg?t=${Date.now()}${full}`;
  };
  img.onload = () => setTimeout(next, 70);
  img.onerror = () => setTimeout(next, 1000);
  next();
}

function flashTile(camId) {
  const tile = document.querySelector(`.tile[data-id="${CSS.escape(camId)}"]`);
  if (!tile) return;
  tile.classList.add("flash");
  clearTimeout(tile._flash);
  tile._flash = setTimeout(() => tile.classList.remove("flash"), 3000);
}

// ===================== alerts =====================

function renderCounts(counts) {
  for (const id of ["#alert-new-count", "#tab-badge"]) {
    const el = $(id);
    el.textContent = counts.new;
    el.classList.toggle("hidden", counts.new === 0);
  }
}

function alertItem(a, fresh = false) {
  const track = a.track_id !== null && a.track_id !== undefined ? ` · #${a.track_id}` : "";
  return `
    <li class="alert-item ${a.status === "new" ? "new" : ""} ${fresh ? "fresh" : ""}" data-id="${esc(a.id)}">
      <img src="/media/alerts/${encodeURI(a.thumb || a.image)}" loading="lazy" alt="" data-act="view" title="Open image">
      <div class="alert-meta">
        <b>${esc(a.label)}</b>
        <span>${esc(a.camera_name)} · ${esc(a.location)}</span>
        <span>${esc(fmtTime(a.time))} · ${Math.round(a.confidence * 100)}%${track}</span>
      </div>
      <div class="alert-actions">
        ${a.status === "new" ? `<button class="icon-btn" data-act="ack" title="Acknowledge">✓</button>` : ""}
        <button class="icon-btn" data-act="del" title="Delete">✕</button>
      </div>
    </li>`;
}

function renderAlerts() {
  $("#alert-list").innerHTML = state.alerts.map((a) => alertItem(a)).join("");
  $("#alert-empty").classList.toggle("hidden", state.alerts.length > 0);
}

async function loadAlerts() {
  const q = state.filterCamera ? `&camera=${encodeURIComponent(state.filterCamera)}` : "";
  try {
    const d = await api("GET", `/api/alerts?limit=200${q}`);
    state.alerts = d.alerts;
    renderAlerts();
    renderCounts(d.counts);
  } catch (err) {
    toast(`Could not load alerts: ${err.message}`, "error");
  }
}

function onNewAlert(a) {
  flashTile(a.camera_id);
  toast(`🚨 ${a.label} - ${a.camera_name}`, "alert", 6000);
  if (state.sound) beep();
  refreshStatus();
  if (state.view === "violations" && vstate.page === 0) loadViolations();
  if (state.filterCamera && state.filterCamera !== a.camera_id) return;
  state.alerts.unshift(a);
  $("#alert-list").insertAdjacentHTML("afterbegin", alertItem(a, true));
  $("#alert-empty").classList.add("hidden");
}

let audioCtx = null;
function beep() {
  try {
    audioCtx = audioCtx || new AudioContext();
    [0, 0.22].forEach((offset) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.frequency.value = 880;
      gain.gain.value = 0.15;
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(audioCtx.currentTime + offset);
      osc.stop(audioCtx.currentTime + offset + 0.15);
    });
  } catch { /* audio not available */ }
}

function openImage(alert) {
  $("#image-dialog-img").src = `/media/alerts/${encodeURI(alert.image)}`;
  $("#image-dialog-caption").innerHTML =
    `<b>${esc(alert.label)}</b> · ${esc(alert.camera_name)} (${esc(alert.location)}) · ` +
    `${esc(fmtTime(alert.time))} · ${Math.round(alert.confidence * 100)}%` +
    `<div class="muted small">data/alerts/${esc(alert.image)}</div>`;
  $("#image-dialog").showModal();
}

// view / ack / del - shared by the dashboard alert panel and the Violations table
async function alertAction(act, alert) {
  if (act === "view") {
    openImage(alert);
    return false;
  }
  if (act === "ack") {
    Object.assign(alert, await api("PATCH", `/api/alerts/${alert.id}`, { status: "acknowledged" }));
  } else if (act === "del") {
    if (!confirm(`Delete this "${alert.label}" violation and its image?`)) return false;
    await api("DELETE", `/api/alerts/${alert.id}`);
  }
  refreshStatus();
  return true;   // data changed
}

$("#alert-list").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const li = btn.closest(".alert-item");
  const alert = state.alerts.find((a) => a.id === li.dataset.id);
  if (!alert) return;
  try {
    if (!(await alertAction(btn.dataset.act, alert))) return;
    if (btn.dataset.act === "ack") {
      li.outerHTML = alertItem(alert);
    } else {
      state.alerts = state.alerts.filter((a) => a.id !== alert.id);
      li.remove();
      $("#alert-empty").classList.toggle("hidden", state.alerts.length > 0);
    }
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#alert-filter-camera").addEventListener("change", (e) => {
  state.filterCamera = e.target.value;
  loadAlerts();
});

$("#ack-all").addEventListener("click", async () => {
  try {
    await api("POST", "/api/alerts/acknowledge-all");
    await loadAlerts();
  } catch (err) { toast(err.message, "error"); }
});

$("#clear-alerts").addEventListener("click", async () => {
  if (!confirm("Delete ALL alerts and their saved images? This cannot be undone.")) return;
  try {
    await api("DELETE", "/api/alerts");
    await loadAlerts();
    toast("All alerts deleted");
  } catch (err) { toast(err.message, "error"); }
});

$("#sound-toggle").checked = state.sound;
$("#sound-toggle").addEventListener("change", (e) => {
  state.sound = e.target.checked;
  writePref("sound", state.sound ? "1" : "0");
  if (state.sound) beep();
});

function connectEvents() {
  const es = new EventSource("/api/alerts/events");
  let dropped = false;
  es.addEventListener("alert", (e) => onNewAlert(JSON.parse(e.data)));
  es.onerror = () => { dropped = true; };
  es.onopen = () => {
    if (dropped) { dropped = false; loadAlerts(); }   // reconnected: catch up on anything missed
  };
}

// ===================== violations page =====================

const vstate = { page: 0, size: 25, total: 0, items: [] };

function violationFilters() {
  const p = new URLSearchParams();
  const add = (key, sel) => { if ($(sel).value) p.set(key, $(sel).value); };
  add("camera", "#v-camera");
  add("class", "#v-type");
  add("status", "#v-status");
  add("from", "#v-from");
  add("to", "#v-to");
  return p;
}

async function loadViolations() {
  const filters = violationFilters();
  const p = new URLSearchParams(filters);
  p.set("limit", vstate.size);
  p.set("offset", vstate.page * vstate.size);
  let d;
  try {
    d = await api("GET", `/api/alerts?${p}`);
  } catch (err) {
    toast(`Could not load violations: ${err.message}`, "error");
    return;
  }
  // page became empty (e.g. after deleting its last row) - step back
  if (!d.alerts.length && vstate.page > 0 && d.total > 0) {
    vstate.page = Math.floor((d.total - 1) / vstate.size);
    return loadViolations();
  }
  vstate.items = d.alerts;
  vstate.total = d.total;
  renderCounts(d.counts);

  const typeSel = $("#v-type");
  const typeOpts = d.types.map((t) => t.name).join("|");
  if (typeSel.dataset.opts !== typeOpts) {
    typeSel.dataset.opts = typeOpts;
    const value = typeSel.value;
    typeSel.innerHTML = `<option value="">All types</option>` +
      d.types.map((t) => `<option value="${esc(t.name)}">${esc(t.label)}</option>`).join("");
    typeSel.value = value;
  }

  const offset = vstate.page * vstate.size;
  $("#v-rows").innerHTML = d.alerts.map((a, i) => {
    const t = new Date(a.time);
    return `
      <tr data-id="${esc(a.id)}" class="${a.status === "new" ? "is-new" : ""}">
        <td class="muted">${offset + i + 1}</td>
        <td><img class="vthumb" src="/media/alerts/${encodeURI(a.thumb || a.image)}" loading="lazy" alt="" data-act="view"></td>
        <td>${esc(t.toLocaleDateString())}<div class="muted small">${esc(t.toLocaleTimeString())}</div></td>
        <td>${esc(a.camera_name)}</td>
        <td>${esc(a.location)}</td>
        <td><b>${esc(a.label)}</b>${a.label !== a.class_name ? `<div class="muted small">${esc(a.class_name)}</div>` : ""}</td>
        <td>${Math.round(a.confidence * 100)}%</td>
        <td>${a.track_id ?? "–"}</td>
        <td><span class="status ${esc(a.status)}">${a.status === "new" ? "New" : "Acknowledged"}</span></td>
        <td class="actions">
          <button class="btn small" data-act="view">View</button>
          ${a.status === "new" ? `<button class="btn small" data-act="ack">Acknowledge</button>` : ""}
          <button class="btn small danger-ghost" data-act="del">Delete</button>
        </td>
      </tr>`;
  }).join("");
  $("#v-empty").classList.toggle("hidden", d.alerts.length > 0);

  const pages = Math.max(1, Math.ceil(d.total / vstate.size));
  const filtered = [...filters.keys()].length > 0;
  $("#v-summary").textContent = filtered
    ? `${d.total} of ${d.counts.total} violations match the filters · ${d.counts.new} new`
    : `${d.counts.total} violations · ${d.counts.new} new`;
  $("#v-page").textContent = d.total
    ? `${offset + 1}–${offset + d.alerts.length} of ${d.total} · page ${vstate.page + 1} of ${pages}`
    : "";
  $("#v-prev").disabled = vstate.page === 0;
  $("#v-next").disabled = vstate.page >= pages - 1;
  $("#v-export").href = `/api/alerts/export.csv?${filters}`;
}

$("#v-rows").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const alert = vstate.items.find((a) => a.id === btn.closest("tr").dataset.id);
  if (!alert) return;
  try {
    if (await alertAction(btn.dataset.act, alert)) {
      await loadViolations();
      loadAlerts();   // keep the dashboard panel in sync
    }
  } catch (err) {
    toast(err.message, "error");
  }
});

["#v-camera", "#v-type", "#v-status", "#v-from", "#v-to"].forEach((sel) =>
  $(sel).addEventListener("change", () => { vstate.page = 0; loadViolations(); }));

$("#v-size").addEventListener("change", (e) => {
  vstate.size = Number(e.target.value);
  vstate.page = 0;
  loadViolations();
});
$("#v-prev").addEventListener("click", () => { vstate.page--; loadViolations(); });
$("#v-next").addEventListener("click", () => { vstate.page++; loadViolations(); });

$("#v-reset").addEventListener("click", () => {
  ["#v-camera", "#v-type", "#v-status", "#v-from", "#v-to"].forEach((sel) => { $(sel).value = ""; });
  vstate.page = 0;
  loadViolations();
});

$("#v-ack-all").addEventListener("click", async () => {
  try {
    const { acknowledged } = await api("POST", "/api/alerts/acknowledge-all");
    toast(`${acknowledged} violation${acknowledged === 1 ? "" : "s"} acknowledged`);
    await loadViolations();
    loadAlerts();
  } catch (err) { toast(err.message, "error"); }
});

// ===================== cameras page =====================

async function loadCameras() {
  try {
    state.cameras = await api("GET", "/api/cameras");
  } catch (err) {
    toast(err.message, "error");
    return;
  }
  $("#cameras-empty").classList.toggle("hidden", state.cameras.length > 0);
  $("#camera-rows").innerHTML = state.cameras.map((c) => `
    <tr data-id="${esc(c.id)}">
      <td><b>${esc(c.name)}</b></td>
      <td>${esc(c.location)}</td>
      <td><span class="tag">${esc(c.type)}</span></td>
      <td class="src">${esc(c.source)}</td>
      <td><input type="checkbox" data-act="toggle" ${c.enabled ? "checked" : ""}></td>
      <td class="actions">
        <button class="btn small" data-act="edit">Edit</button>
        <button class="btn small danger-ghost" data-act="del">Delete</button>
      </td>
    </tr>`).join("");
}

function savedMessage(what, res) {
  toast(`${what} saved${res.restarted ? " - detection restarted" : ""}`);
}

$("#camera-rows").addEventListener("click", async (e) => {
  const el = e.target.closest("[data-act]");
  if (!el) return;
  const cam = state.cameras.find((c) => c.id === el.closest("tr").dataset.id);
  try {
    if (el.dataset.act === "edit") {
      openCameraDialog(cam);
    } else if (el.dataset.act === "del") {
      if (!confirm(`Delete camera "${cam.name}"? Its saved alerts are kept.`)) return;
      const res = await api("DELETE", `/api/cameras/${cam.id}`);
      toast(`Camera deleted${res.restarted ? " - detection restarted" : ""}`);
      await loadCameras();
      refreshStatus();
    } else if (el.dataset.act === "toggle") {
      const res = await api("PUT", `/api/cameras/${cam.id}`, { enabled: el.checked });
      savedMessage("Camera", res);
      await loadCameras();
      refreshStatus();
    }
  } catch (err) {
    toast(err.message, "error");
    loadCameras();
  }
});

$("#add-camera").addEventListener("click", () => openCameraDialog(null));

function openCameraDialog(cam) {
  const form = $("#camera-form");
  state.editingCameraId = cam ? cam.id : null;
  $("#camera-dialog-title").textContent = cam ? `Edit camera - ${cam.name}` : "Add camera";
  form.name.value = cam?.name ?? "";
  form.location.value = cam?.location ?? "";
  form.type.value = cam?.type ?? "video";
  form.source.value = cam?.source ?? "";
  form.enabled.checked = cam ? cam.enabled : true;
  form.source.placeholder = SOURCE_HINTS[form.type.value];
  $("#camera-test-result").className = "test-result hidden";
  $("#camera-dialog").showModal();
}

function cameraFormData() {
  const form = $("#camera-form");
  return {
    name: form.name.value.trim(),
    location: form.location.value.trim(),
    type: form.type.value,
    source: form.source.value.trim(),
    enabled: form.enabled.checked,
  };
}

$("#camera-form").type.addEventListener("change", (e) => {
  $("#camera-form").source.placeholder = SOURCE_HINTS[e.target.value];
});

$("#camera-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = cameraFormData();
  try {
    const res = state.editingCameraId
      ? await api("PUT", `/api/cameras/${state.editingCameraId}`, data)
      : await api("POST", "/api/cameras", data);
    $("#camera-dialog").close();
    savedMessage("Camera", res);
    await loadCameras();
    refreshStatus();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#camera-test").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  const box = $("#camera-test-result");
  btn.disabled = true;
  box.className = "test-result";
  box.textContent = "Connecting…";
  try {
    const res = await api("POST", "/api/cameras/test", cameraFormData());
    box.className = `test-result ${res.ok ? "ok" : "bad"}`;
    box.innerHTML = esc(res.message) + (res.image ? `<img src="${res.image}" alt="">` : "");
  } catch (err) {
    box.className = "test-result bad";
    box.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

// ===================== model page =====================

const getPath = (obj, path) => path.split(".").reduce((o, k) => o?.[k], obj);
function setPath(obj, path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  keys.reduce((o, k) => (o[k] ??= {}), obj)[last] = value;
}

function syncRangeOutputs() {
  $$("#model-form .range").forEach((r) => {
    $("output", r).textContent = Number($("input", r).value).toFixed(2);
  });
}
$("#model-form").addEventListener("input", (e) => {
  if (e.target.type === "range") syncRangeOutputs();
});

function fillModel(model, warnings) {
  const form = $("#model-form");
  $$("[name]", form).forEach((el) => {
    const v = getPath(model, el.name);
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v ?? "";
  });
  syncRangeOutputs();
  renderClassRows(model.classes);
  $("#model-warnings").innerHTML = warnings.map((w) => `<div class="warning">⚠ ${esc(w)}</div>`).join("");
}

async function loadModel() {
  try {
    const d = await api("GET", "/api/model");
    fillModel(d.model, d.warnings);
  } catch (err) {
    toast(err.message, "error");
  }
}

function classRow(c) {
  return `
    <tr>
      <td><input type="text" data-k="name" value="${esc(c.name)}" placeholder="class name"></td>
      <td><input type="color" data-k="color" value="${esc(c.color)}"></td>
      <td><input type="checkbox" data-k="show" ${c.show ? "checked" : ""}></td>
      <td><input type="checkbox" data-k="alert" ${c.alert ? "checked" : ""}></td>
      <td><input type="text" data-k="alert_label" value="${esc(c.alert_label)}" placeholder="shown on alert"></td>
      <td class="actions"><button type="button" class="icon-btn" data-act="remove" title="Remove">✕</button></td>
    </tr>`;
}

function renderClassRows(classes) {
  $("#class-rows").innerHTML = classes.length
    ? classes.map(classRow).join("")
    : `<tr><td colspan="6" class="muted">No classes yet - click <b>Load classes from model file</b>.</td></tr>`;
}

function readClassRows() {
  return $$("#class-rows tr").filter((tr) => $("[data-k=name]", tr)).map((tr) => ({
    name: $("[data-k=name]", tr).value.trim(),
    color: $("[data-k=color]", tr).value,
    show: $("[data-k=show]", tr).checked,
    alert: $("[data-k=alert]", tr).checked,
    alert_label: $("[data-k=alert_label]", tr).value.trim(),
  })).filter((c) => c.name);
}

$("#class-rows").addEventListener("click", (e) => {
  if (e.target.closest("[data-act=remove]")) e.target.closest("tr").remove();
});

$("#add-class").addEventListener("click", () => {
  const current = readClassRows();
  current.push({ name: "", color: PALETTE[current.length % PALETTE.length], show: true, alert: false, alert_label: "" });
  renderClassRows(current);
  $$("#class-rows [data-k=name]").pop().focus();
});

$("#load-classes").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = "Reading model…";
  try {
    const { classes } = await api("POST", "/api/model/classes", { path: $("#model-form").path.value.trim() });
    const existing = Object.fromEntries(readClassRows().map((c) => [c.name, c]));
    renderClassRows(classes.map((name, i) => existing[name] ?? {
      name, color: PALETTE[i % PALETTE.length], show: true, alert: false, alert_label: name,
    }));
    toast(`Loaded ${classes.length} classes - tick the alert classes, then Save`);
  } catch (err) {
    toast(err.message, "error", 7000);
  } finally {
    btn.disabled = false;
    btn.textContent = "Load classes from model file";
  }
});

$("#model-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const model = {};
  $$("[name]", e.target).forEach((el) => {
    const value = el.type === "checkbox" ? el.checked
      : el.type === "number" || el.type === "range" ? Number(el.value)
      : el.value.trim();
    setPath(model, el.name, value);
  });
  model.classes = readClassRows();
  try {
    const res = await api("PUT", "/api/model", model);
    fillModel(res.model, res.warnings);
    savedMessage("Model settings", res);
    refreshStatus();
  } catch (err) {
    toast(err.message, "error", 7000);
  }
});

// ===================== start =====================

route();
refreshStatus();
setInterval(refreshStatus, 2000);
loadAlerts();
connectEvents();
