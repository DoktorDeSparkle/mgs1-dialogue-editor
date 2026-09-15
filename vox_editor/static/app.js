"use strict";

// ------------------------------------------------------------------ config
const ROW_WIDTH = 40;   // USA subtitle rows: p95 39, p99 42 chars
const MAX_ROWS = 2;     // auto-split target; >3 rows gets flagged
const STRONG_SIM = 0.45; // LaBSE line sim considered a confident line alignment
const AGREE_WARN = 0.7;  // LaBSE sim between engines' English; sampled: >=0.72 paraphrase, 0.6-0.7 real meaning differences

// ------------------------------------------------------------------ globals
let D = null;                 // server data
const CAND = new Map();       // jpn_key -> candidates row
const DUAL = new Map();       // usa_key -> dual-score row
const HUNG = new Map();       // jpn_key -> hungarian row
let STATE = { convs: {} };
let listKeys = [];            // keys in the sidebar after filter/sort
let cur = null;               // current jpn key
let conv = null;              // STATE.convs[cur] (possibly unsaved new)
let focusIdx = 0;             // focused chunk index
let selLines = [];            // selected line keys (in order)
let selAnchor = null;
let shiftHeld = false;        // fallback for clicks whose event lacks shiftKey
let ctxAt = -Infinity;        // last macOS Ctrl-click (contextmenu) toggle, so a following click doesn't undo it
let undoStack = [], redoStack = [];
let subTarget = null;         // subtitle object that → on a USA line replaces (last subtitle clicked into)
const pending = new Map();    // chunk object being translated -> {engine: "wait" | "ok" | "fail"}
let translateAllRun = null;   // {stop: bool}

const $ = (s) => document.querySelector(s);
const IS_MAC = /Mac/.test(navigator.platform);
const DS = new URLSearchParams(location.search).get("ds") || "vox"; // dataset from vox_editor/datasets.json

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
  return el;
}

function toast(msg, err = false, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg; t.className = err ? "err" : ""; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}

async function api(path, body) {
  const r = body
    ? await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ds: DS, ...body }) })
    : await fetch(`${path}?ds=${encodeURIComponent(DS)}`);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

// ------------------------------------------------------------------ text helpers
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const stripTags = (s) => s.replace(/‹[A-Z]+›/g, "");

// JPN markup -> HTML with <ruby> for ＃｛base、reading｝＃ blocks
function jpnHtml(raw) {
  let s = esc(stripTags(raw));
  s = s.replace(/[#＃]｛([^｝]*)｝[#＃]?/g, (_, inner) => { // vox uses ＃, demo uses #
    const [base, ...rest] = inner.split("、");
    return rest.length ? `<ruby>${base}<rt>${rest.join("、")}</rt></ruby>` : base;
  });
  return s.replace(/[#＃｛｝]/g, "").replace(/｜/g, "\n").trim();
}
// JPN markup -> plain text for MT: base（reading）
function jpnPlain(raw) {
  let s = stripTags(raw).replace(/[#＃]｛([^｝]*)｝[#＃]?/g, (_, inner) => {
    const [base, ...rest] = inner.split("、");
    // a Latin base (FOX HOUND) is already the term; a kanji base keeps its reading (漢字（かんじ）)
    return rest.length && !/[A-Za-z]/.test(base) ? `${base}（${rest.join("、")}）` : base;
  });
  return s.replace(/[#＃｛｝]/g, "").replace(/[｜\r\n]/g, " ").trim();
}
// US script line breaks are \r (sometimes followed by tab indentation, e.g. vox-0078) or ｜
const usaText = (raw) => raw.replace(/\r\n?|｜/g, "\n").replace(/[ \t]*\n[ \t]*/g, "\n").replace(/\t/g, " ").trim();

const parseTiming = (s) => { const [a, b] = String(s || "0,0").split(",").map(Number); return [a || 0, b || 0]; };
const lineKeysOf = (entry) => Object.keys(entry?.[0] || {}).sort();
const nonEmptyCount = (entry) => lineKeysOf(entry).filter((k) => jpnPlain(entry[0][k])).length;

// Wrap one subtitle into rows <= ROW_WIDTH, preferring a balanced 2-row break.
function wrapRows(text) {
  text = text.replace(/\s+/g, " ").trim();
  if (text.length <= ROW_WIDTH) return text;
  if (text.length <= ROW_WIDTH * 2) {
    const mid = text.length / 2;
    let best = -1;
    for (let i = 0; i < text.length; i++) {
      if (text[i] !== " ") continue;
      if (i > ROW_WIDTH || text.length - i - 1 > ROW_WIDTH) continue;
      if (best < 0 || Math.abs(i - mid) < Math.abs(best - mid)) best = i;
    }
    if (best > 0) return text.slice(0, best) + "\n" + text.slice(best + 1);
  }
  const rows = []; let row = "";
  for (const w of text.split(" ")) {
    if (row && (row + " " + w).length > ROW_WIDTH) { rows.push(row); row = w; } else row = row ? row + " " + w : w;
  }
  if (row) rows.push(row);
  return rows.join("\n");
}

// Split a translated chunk into subtitle-sized pieces (<= MAX_ROWS rows each).
function splitPieces(text) {
  text = text.replace(/\s+/g, " ").trim();
  const cap = ROW_WIDTH * MAX_ROWS;
  if (text.length <= cap) return [text];
  const sentences = text.split(/(?<=[.!?…—])\s+/);
  const pieces = []; let acc = "";
  for (let s of sentences) {
    while (s.length > cap) { // overlong sentence: cut at a space near the cap, prefer after a comma
      let cut = s.lastIndexOf(", ", cap); if (cut < cap * 0.4) cut = s.lastIndexOf(" ", cap);
      if (cut <= 0) cut = cap;
      if (acc) { pieces.push(acc); acc = ""; }
      pieces.push(s.slice(0, cut + (s[cut] === "," ? 1 : 0)).trim()); s = s.slice(cut + 1).trim();
    }
    if (acc && (acc + " " + s).length > cap) { pieces.push(acc); acc = s; } else acc = acc ? acc + " " + s : s;
  }
  if (acc) pieces.push(acc);
  return pieces;
}

function chunkSpan(chunk, key = cur) {
  const tm = D.jpn[key][1];
  let start = Infinity, end = 0;
  for (const k of chunk.lines) { const [s, d] = parseTiming(tm[k]); start = Math.min(start, s); end = Math.max(end, s + d); }
  return start === Infinity ? [0, 0] : [start, end];
}

// Assign timings proportional to character count across the chunk span.
function redistribute(chunk, key = cur) {
  const [start, end] = chunkSpan(chunk, key);
  const total = chunk.subs.reduce((n, s) => n + Math.max(s.text.length, 1), 0) || 1;
  let t = start;
  chunk.subs.forEach((s, i) => {
    const d = i === chunk.subs.length - 1 ? end - t : Math.round(((end - start) * Math.max(s.text.length, 1)) / total);
    s.start = Math.round(t); s.dur = Math.max(1, Math.round(d)); t += d;
  });
}

function autoSubs(chunk, text, key = cur) {
  chunk.subs = splitPieces(text).map((p) => ({ text: wrapRows(p), start: 0, dur: 0 }));
  chunk.timingEdited = false; chunk.subsEdited = false;
  redistribute(chunk, key);
}

// ------------------------------------------------------------------ match confidence
function confidence(jk) {
  const row = CAND.get(jk);
  const c = row?.candidates || [];
  if (!c.length) return { level: "none", score: -1 };
  const top = c[0], d = DUAL.get(top.usa_key);
  const mutual = !!d && d.line_best_jpn === jk && d.whole_best_jpn === jk;
  const hung = HUNG.get(jk)?.usa_key === top.usa_key;
  const margin = top.combined_score - (c[1]?.combined_score ?? 0);
  let level = "low";
  if (mutual && top.combined_score >= 0.7) level = "high";
  else if (mutual || hung || (top.combined_score >= 0.6 && margin >= 0.1)) level = "med";
  return { level, score: top.combined_score + (mutual ? 0.2 : 0) + (hung ? 0.1 : 0) + margin * 0.5, mutual, hung, margin };
}

// ------------------------------------------------------------------ state / persistence
const STATUS_LABEL = { todo: "To do", mt: "MT draft", wip: "In progress", done: "Done" };

function newConv(jk) {
  return {
    status: "todo",
    usa_ref: (D.same_id && lineKeysOf(D.usa[jk]).length ? jk : null) || CAND.get(jk)?.candidates?.[0]?.usa_key || HUNG.get(jk)?.usa_key || null,
    chunks: lineKeysOf(D.jpn[jk]).map((k) => ({ lines: [k], mt: "", subs: [] })),
  };
}

let saveTimer = null;
function touch({ structural = false } = {}) {
  if (conv.status === "todo" || conv.status === "mt") { conv.status = "wip"; $("#statusSel").value = "wip"; }
  STATE.convs[cur] = conv;
  $("#saveState").textContent = "unsaved…";
  clearTimeout(saveTimer);
  const key = cur, c = conv;
  saveTimer = setTimeout(() => save(key, c), 600);
  if (structural) renderList();
}
async function save(key, c) {
  try { c.updated = new Date().toISOString(); await api("/api/save", { key, conv: c }); $("#saveState").textContent = "saved"; }
  catch (e) { $("#saveState").textContent = "SAVE FAILED"; toast("Save failed: " + e.message, true); }
}

const snapshot = () => JSON.stringify(conv);
function pushUndo(snap = snapshot()) { undoStack.push(snap); if (undoStack.length > 200) undoStack.shift(); redoStack = []; }
function restore(snap) { conv = JSON.parse(snap); STATE.convs[cur] = conv; focusIdx = Math.min(focusIdx, conv.chunks.length - 1); touch({ structural: true }); renderEditor(); renderRef(); }
function undo() { if (!undoStack.length) return; redoStack.push(snapshot()); restore(undoStack.pop()); }
function redo() { if (!redoStack.length) return; undoStack.push(snapshot()); restore(redoStack.pop()); }
// wrap a structural mutation: snapshot, mutate, save, re-render
function mutate(fn) { pushUndo(); fn(); touch({ structural: true }); renderEditor(); renderRef(); }

// ------------------------------------------------------------------ sidebar
function renderList() {
  const q = $("#search").value.trim().toLowerCase();
  const minLines = +$("#minLines").value, st = $("#statusFilter").value, cf = $("#confFilter").value, sort = $("#sortBy").value;
  const rows = [];
  for (const jk of Object.keys(D.jpn)) {
    const n = nonEmptyCount(D.jpn[jk]);
    if (n < minLines) continue;
    const status = STATE.convs[jk]?.status || "todo";
    if (st !== "all" && status !== st) continue;
    const conf = confidence(jk);
    if (cf !== "all" && conf.level !== cf) continue;
    if (q && !jk.includes(q)) {
      const hay = Object.values(D.jpn[jk][0]).join(" ") + JSON.stringify(STATE.convs[jk]?.chunks || "");
      if (!hay.toLowerCase().includes(q)) continue;
    }
    rows.push({ jk, n, status, conf });
  }
  if (sort === "conf-desc") rows.sort((a, b) => b.conf.score - a.conf.score);
  if (sort === "conf-asc") rows.sort((a, b) => a.conf.score - b.conf.score);
  if (sort === "lines-desc") rows.sort((a, b) => b.n - a.n);
  if (sort === "agree-asc") rows.sort((a, b) => (STATE.convs[a.jk]?.mt_agree ?? 2) - (STATE.convs[b.jk]?.mt_agree ?? 2));
  listKeys = rows.map((r) => r.jk);

  const ul = $("#convList"); ul.replaceChildren();
  for (const r of rows) {
    const first = lineKeysOf(D.jpn[r.jk]).map((k) => jpnPlain(D.jpn[r.jk][0][k])).find(Boolean) || "";
    const top = CAND.get(r.jk)?.candidates?.[0];
    const agree = STATE.convs[r.jk]?.mt_agree;
    ul.append(h("li", { class: r.jk === cur ? "active" : "", "data-key": r.jk, onclick: () => openConv(r.jk) },
      h("span", { class: `dot ${r.status}`, title: STATUS_LABEL[r.status] }),
      h("span", { class: "key" }, r.jk, h("span", { class: "muted small" }, `  ${r.n}L`),
        agree != null && agree < AGREE_WARN && h("span", { class: "agree-flag", title: `lowest MT agreement between engines: ${agree}` }, ` ≠${agree.toFixed(2)}`)),
      h("span", { class: `conf ${r.conf.level}`, title: top ? `top: ${top.usa_key} ${top.combined_score}` : "no candidate" },
        top ? top.combined_score.toFixed(2) : "—"),
      h("span", { class: "preview" }, first)));
  }
  $("#convCount").textContent = `${rows.length} conversations`;
  const all = Object.values(STATE.convs);
  const count = (st) => all.filter((c) => c.status === st).length;
  $("#progress").textContent = `${count("done")} done · ${count("wip")} in progress · ${count("mt")} MT drafts`;
  ul.querySelector("li.active")?.scrollIntoView({ block: "nearest" });
}

// ------------------------------------------------------------------ conversation
function openConv(jk) {
  if (!D.jpn[jk]) return;
  clearTimeout(saveTimer); if (cur && conv && $("#saveState").textContent === "unsaved…") save(cur, conv);
  cur = jk;
  conv = STATE.convs[jk] ? STATE.convs[jk] : newConv(jk);
  focusIdx = 0; selLines = []; selAnchor = null; undoStack = []; redoStack = [];
  history.replaceState(null, "", "#" + jk);
  $("#statusSel").value = conv.status;
  document.querySelectorAll("#convList li").forEach((li) => li.classList.toggle("active", li.dataset.key === jk));
  renderEditor(); renderRef();
  $("#editorPane").scrollTop = 0;
}

function allLineKeys() { return conv.chunks.flatMap((c) => c.lines); }
const chunkOfLine = (k) => conv.chunks.findIndex((c) => c.lines.includes(k));

// click = select one line, Shift-click = range from the anchor, ⌘/Ctrl-click = toggle one line
// keys: one line key, or a whole chunk's lines (click on the chunk box outside a line)
function clickLine(keys, ev) {
  keys = [].concat(keys);
  const order = allLineKeys(), idx = keys.map((k) => order.indexOf(k));
  if ((ev.shiftKey || shiftHeld) && order.includes(selAnchor)) {
    const at = order.indexOf(selAnchor);
    selLines = order.slice(Math.min(at, ...idx), Math.max(at, ...idx) + 1);
  } else if (ev.metaKey || ev.ctrlKey) {
    const all = keys.every((k) => selLines.includes(k));
    selLines = all ? selLines.filter((x) => !keys.includes(x)) : order.filter((x) => keys.includes(x) || selLines.includes(x));
    selAnchor = keys[0];
  } else { selLines = keys; selAnchor = keys[0]; }
  setFocus(chunkOfLine(keys[0]));
  renderSel();
}
function clearSel() { selLines = []; selAnchor = null; renderSel(); }
function renderSel() {
  document.querySelectorAll(".jline").forEach((el) => el.classList.toggle("selected", selLines.includes(el.dataset.line)));
  const spans = new Set(selLines.map(chunkOfLine)).size;
  $("#selInfo").hidden = !selLines.length;
  $("#selInfo").textContent = `${selLines.length} selected` + (spans > 1 ? " · G" : "");
  $("#selInfo").title = (spans > 1 ? "G / Group joins these into one chunk" : "Shift-click a range or ⌘/Ctrl-click lines to add") + " · Esc clears";
  $("#groupBtn").classList.toggle("primary", spans > 1);
}

function setFocus(i, scroll = false) {
  if (i < 0 || i >= conv.chunks.length) return;
  focusIdx = i;
  document.querySelectorAll(".chunk").forEach((el, j) => el.classList.toggle("focused", j === i));
  if (scroll) document.querySelectorAll(".chunk")[i]?.scrollIntoView({ block: "nearest" });
  highlightRef(true);
}

function groupSelection() {
  if (selLines.length < 2) { toast("Select lines first: click one, then Shift-click (range) or ⌘/Ctrl-click (add). Or use “join ↓” on a chunk."); return; }
  const idx = selLines.map(chunkOfLine);
  const a = Math.min(...idx), b = Math.max(...idx);
  if (a === b) { toast("Those lines are already one chunk"); return; }
  mergeChunks(a, b);
}
// Regrouping changes what the engines see, so merged/split chunks start with no MT (mts, mt_agree, decided
// dropped) and get re-translated. Old subtitles stay until the new MT replaces them; edited ones are kept.
function mergeChunks(a, b) {
  let merged;
  mutate(() => {
    const parts = conv.chunks.slice(a, b + 1);
    merged = {
      lines: parts.flatMap((c) => c.lines),
      mt: "",
      subs: parts.flatMap((c) => c.subs),
      timingEdited: parts.some((c) => c.timingEdited),
      subsEdited: parts.some((c) => c.subsEdited),
    };
    conv.chunks.splice(a, b - a + 1, merged);
    focusIdx = a; selLines = []; selAnchor = null;
  });
  retranslate([merged], merged.subsEdited || merged.timingEdited);
}
function ungroup(i = focusIdx) {
  const c = conv.chunks[i];
  if (!c || c.lines.length < 2) { toast("Focused chunk is a single line"); return; }
  let parts;
  mutate(() => {
    const tm = D.jpn[cur][1];
    parts = c.lines.map((k) => ({ lines: [k], mt: "", subs: [], timingEdited: c.timingEdited, subsEdited: c.subsEdited }));
    // give each subtitle to the line whose window contains its start (else the last line that started before it)
    for (const s of c.subs) {
      let target = 0;
      c.lines.forEach((k, j) => { if (parseTiming(tm[k])[0] <= s.start) target = j; });
      parts[target].subs.push(s);
    }
    conv.chunks.splice(i, 1, ...parts);
  });
  retranslate(parts, c.subsEdited || c.timingEdited);
}
// one chunk at a time (keeps LM Studio load down); no undo steps, so one undo reverts the regroup
async function retranslate(chunks, keepSubs) {
  const key = cur;
  for (const c of chunks) {
    const i = conv.chunks.indexOf(c);
    if (cur !== key || i < 0) break;
    try { await translateChunk(i, { undo: false, keepSubs }); } catch { break; }
  }
}

async function translateChunk(i, { undo = true, keepSubs = false } = {}) {
  const chunk = conv.chunks[i];
  if (!chunk || pending.has(chunk)) return;
  const key = cur, owner = conv, entry = D.jpn[key];
  const text = chunk.lines.map((k) => jpnPlain(entry[0][k])).join("");
  if (!text) return;
  const context = conv.chunks.map((c) => (c === chunk ? "▶ " : "  ") + c.lines.map((k) => jpnPlain(entry[0][k])).join("")).join("\n");
  const ref = conv.usa_ref && D.usa[conv.usa_ref] ? lineKeysOf(D.usa[conv.usa_ref]).map((k) => usaText(D.usa[conv.usa_ref][0][k]).replace(/\n/g, " ")).join("\n") : "";
  const st = Object.fromEntries(D.engines.map((e) => [e.name, "wait"]));
  pending.set(chunk, st); renderEditor(); updatePending(chunk);
  try {
    // every configured engine runs in parallel; the primary's output drives mt + subtitles
    const results = await Promise.allSettled(D.engines.map((e) => api("/api/translate", { text, context, reference: ref, engine: e.name })
      .then((r) => { st[e.name] = "ok"; updatePending(chunk); return r; }, (err) => { st[e.name] = "fail"; updatePending(chunk); throw err; })));
    const failed = results.filter((r) => r.status === "rejected");
    if (failed.length === results.length) throw failed[0].reason;
    if (failed.length) toast(`${failed.length} engine(s) failed: ${failed[0].reason.message}`, true, 7000);
    if (!(key === cur ? conv : owner).chunks.includes(chunk)) return; // undone/regrouped meanwhile
    if (key === cur && undo) pushUndo();
    chunk.mts = { ...(chunk.mts || {}) };
    results.forEach((r) => { if (r.status === "fulfilled") chunk.mts[r.value.engine] = r.value.text; });
    delete chunk.mt_agree;
    const best = chunk.mts[D.primary] ?? Object.values(chunk.mts)[0];
    chunk.mt = best;
    if (!keepSubs && (!chunk.subs.length || !(chunk.subsEdited || chunk.timingEdited))) autoSubs(chunk, best, key);
    if (key !== cur) { // user navigated away; apply to stored conv directly
      if (owner.status === "todo" || owner.status === "mt") owner.status = "wip";
      STATE.convs[key] = owner; save(key, owner);
      return;
    }
    touch();
  } catch (e) {
    toast(`Translate failed: ${e.message}`, true, 7000);
    throw e;
  } finally {
    pending.delete(chunk); updatePending(chunk);
    if (key === cur) { renderEditor(); highlightRef(false); }
  }
}

// blinking "translating" light per chunk (with per-engine progress) and in the top bar
function pendingBar(chunk) {
  const st = pending.get(chunk) || {};
  return h("div", { class: "pending-bar" }, h("span", { class: "blink" }), "Translating…",
    Object.entries(st).map(([name, s]) => h("span", { class: `eng-st ${s}` }, `${shortEngine(name)} ${s === "ok" ? "✓" : s === "fail" ? "✕" : "…"}`)));
}
function updatePending(chunk) {
  const n = pending.size;
  $("#mtBusy").hidden = !n;
  $("#mtBusy").lastElementChild.textContent = `Translating ${n} chunk${n === 1 ? "" : "s"}`;
  const i = conv?.chunks.indexOf(chunk) ?? -1;
  const bar = i >= 0 && document.querySelectorAll(".chunk")[i]?.querySelector(".pending-bar");
  if (bar && pending.has(chunk)) bar.replaceWith(pendingBar(chunk));
}

async function translateAll() {
  if (translateAllRun) { translateAllRun.stop = true; return; }
  const run = (translateAllRun = { stop: false });
  const key = cur;
  $("#translateAllBtn").textContent = "Stop";
  try {
    for (let i = 0; i < conv.chunks.length; i++) {
      if (run.stop || cur !== key) break;
      if (conv.chunks[i].mt) continue;
      setFocus(i, true);
      try { await translateChunk(i); } catch { break; }
    }
  } finally { translateAllRun = null; $("#translateAllBtn").textContent = "Translate all"; }
}

// ------------------------------------------------------------------ editor render
function renderEditor() {
  const entry = D.jpn[cur], box = $("#chunks");
  const top = CAND.get(cur)?.candidates?.[0];
  $("#jpnTitle").textContent = `JPN ${cur}`;
  const nSubs = conv.chunks.reduce((n, c) => n + c.subs.length, 0);
  const missing = conv.chunks.filter((c) => !c.subs.length).length;
  $("#jpnMeta").textContent = `${lineKeysOf(entry).length} lines · ${conv.chunks.length} chunks · ${nSubs} subtitles` +
    (missing ? ` · ${missing} chunks without subtitles (export keeps JPN)` : "") + (top ? "" : " · no vector candidate");
  $("#undoBtn").disabled = !undoStack.length; $("#redoBtn").disabled = !redoStack.length;

  box.replaceChildren();
  if (!conv.chunks.length) { box.append(h("div", { class: "placeholder" }, "Empty conversation.")); return; }
  conv.chunks.forEach((chunk, i) => box.append(renderChunk(chunk, i, entry)));
  renderSel();
}

function renderChunk(chunk, i, entry) {
  const [spanS, spanE] = chunkSpan(chunk);
  const src = h("div", { class: "src" },
    chunk.lines.map((k) => {
      const [s, d] = parseTiming(entry[1][k]);
      return h("div", { class: "jline" + (selLines.includes(k) ? " selected" : ""), "data-line": k, title: entry[0][k] },
        h("span", { class: "no" }, k), h("span", { class: "jt", html: jpnHtml(entry[0][k]) || "<span class=muted>(empty)</span>" }),
        h("span", { class: "tm" }, `${s} +${d}`));
    }),
    h("div", { class: "chunk-tools" },
      h("span", { class: "muted small" }, `${spanS}–${spanE}`), h("span", { class: "spacer" }),
      chunk.lines.length > 1 && h("button", { class: "tiny", title: "Split back into single lines and re-translate each (U)", onclick: () => ungroup(i) }, "ungroup"),
      i < conv.chunks.length - 1 && h("button", { class: "tiny join", title: "Group with the next chunk and re-translate", onclick: () => mergeChunks(i, i + 1) }, "join ↓"),
      h("button", { class: "tiny", onclick: () => { setFocus(i); translateChunk(i).catch(() => {}); } },
        pending.has(chunk) ? h("span", { class: "spin" }, "◌") : chunk.mt ? "re-translate" : "translate")));

  const tgt = h("div", { class: "tgt" });
  if (pending.has(chunk)) tgt.append(pendingBar(chunk));
  // engine suggestions; "use" puts one into the subtitles, which are the only editable text
  const mts = Object.entries(chunk.mts || {});
  if (mts.length) {
    const low = chunk.mt_agree != null && chunk.mt_agree < AGREE_WARN;
    const inSubs = norm(chunk.subs.map((s) => s.text).join(" "));
    tgt.append(h("div", { class: "alts" + (low ? " low" : "") },
      low && h("div", { class: "agree-note" }, `Engines disagree (similarity ${chunk.mt_agree.toFixed(2)}) — check the Japanese`),
      mts.map(([name, text]) => {
        const chosen = norm(text) === inSubs;
        return h("div", { class: "alt" + (chosen ? " chosen" : "") },
          h("span", { class: "eng", title: name }, shortEngine(name)),
          h("span", { class: "alt-text" }, text),
          h("button", { class: "tiny", title: chunk.subs.length > 1 && chunk.subs.includes(subTarget)
              ? "Replace the text of the marked subtitle (timing kept)" : "Put into the subtitles, auto-split over the JPN timing",
            onclick: () => useMt(chunk, name, text) }, chosen ? "in subs" : "use"));
      })));
  }

  if (!chunk.subs.length) tgt.append(h("div", { class: "empty" }, "No subtitles yet — translate, type, or copy a USA line (→)."));
  chunk.subs.forEach((sub, j) => tgt.append(renderSub(chunk, i, sub, j)));

  const tl = h("div", { class: "timeline", title: "subtitle timing within this chunk's JPN span" });
  const span = Math.max(1, spanE - spanS);
  let prevEnd = -Infinity;
  for (const s of chunk.subs) {
    const bad = s.start < prevEnd || s.start < spanS || s.start + s.dur > spanE + 1;
    prevEnd = s.start + s.dur;
    tl.append(h("span", { class: bad ? "bad" : "", style: `left:${((s.start - spanS) / span) * 100}%;width:${(s.dur / span) * 100}%` }));
  }
  tgt.append(tl, h("div", { class: "row" },
    h("button", { class: "tiny", onclick: () => mutate(() => { chunk.subs.push({ text: "", start: spanE, dur: 0 }); chunk.subsEdited = true; if (!chunk.timingEdited) redistribute(chunk); }) }, "+ subtitle"),
    chunk.subs.length > 0 && h("button", { class: "tiny", title: "Re-time subtitles proportionally to text length", onclick: () => mutate(() => { redistribute(chunk); chunk.timingEdited = false; }) }, "re-time")));

  const el = h("div", { class: "chunk" + (i === focusIdx ? " focused" : "") + (chunk.lines.length > 1 ? " grouped" : "") + (pending.has(chunk) ? " pending" : "") }, src, tgt);
  // a click on a line selects that line; anywhere else in the box (outside controls) selects the whole chunk
  const target = (ev) => ev.target.closest("textarea, input, select, button") ? null : ev.target.closest(".jline")?.dataset.line ?? chunk.lines;
  el.addEventListener("mousedown", (ev) => {
    if (focusIdx !== i) setFocus(i);
    if ((ev.shiftKey || ev.metaKey || ev.ctrlKey) && target(ev)) ev.preventDefault(); // no text-selection extension
  });
  el.addEventListener("click", (ev) => { const t = target(ev); if (t && !(IS_MAC && ev.ctrlKey && performance.now() - ctxAt < 1000)) clickLine(t, ev); });
  el.addEventListener("contextmenu", (ev) => { const t = target(ev); if (t && ev.ctrlKey && IS_MAC) { ev.preventDefault(); ctxAt = performance.now(); clickLine(t, ev); } }); // macOS Ctrl-click fires contextmenu, usually not click
  return el;
}

const shortEngine = (name) => name.replace(/^google\//, "").split(/[-/]/)[0];

function renderSub(chunk, i, sub, j) {
  const ta = h("textarea", { rows: Math.max(2, sub.text.split("\n").length), spellcheck: "true" });
  ta.value = sub.text;
  const counts = h("div", { class: "counts" });
  const updateCounts = () => {
    const rows = ta.value.split("\n");
    counts.innerHTML = rows.map((r) => `<span class="${r.length > ROW_WIDTH ? "over" : ""}">${r.length}</span>`).join(" · ") +
      (rows.length > 3 ? ' <span class="over">too many rows</span>' : "");
  };
  updateCounts();
  let snap;
  ta.addEventListener("focus", () => {
    snap = snapshot(); setFocus(i);
    subTarget = sub; document.querySelectorAll(".sub.target").forEach((el) => el.classList.remove("target")); ta.parentElement.classList.add("target");
  });
  ta.addEventListener("input", () => {
    sub.text = ta.value; chunk.subsEdited = true; updateCounts(); touch();
    ta.rows = Math.max(2, ta.value.split("\n").length);
  });
  ta.addEventListener("change", () => pushUndo(snap));
  ta.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      const pos = ta.selectionStart, a = ta.value.slice(0, pos).trim(), b = ta.value.slice(pos).trim();
      if (!a || !b) return;
      pushUndo(snap);
      const frac = a.length / (a.length + b.length), d1 = Math.round(sub.dur * frac);
      chunk.subs.splice(j, 1, { text: a, start: sub.start, dur: d1 }, { text: b, start: sub.start + d1, dur: sub.dur - d1 });
      chunk.subsEdited = true; touch(); renderEditor();
      document.querySelectorAll(".chunk")[i]?.querySelectorAll(".sub textarea")[j + 1]?.focus();
    }
  });

  const num = (field) => {
    const inp = h("input", { type: "number", min: 0, value: sub[field], title: field });
    inp.addEventListener("focus", () => (snap = snapshot()));
    inp.addEventListener("change", () => { pushUndo(snap); sub[field] = Math.max(0, Math.round(+inp.value || 0)); chunk.timingEdited = true; touch(); renderEditor(); });
    return inp;
  };
  const move = (dir) => mutate(() => { const k = j + dir; if (k < 0 || k >= chunk.subs.length) return; [chunk.subs[j].text, chunk.subs[k].text] = [chunk.subs[k].text, chunk.subs[j].text]; chunk.subsEdited = true; });
  return h("div", { class: "sub" + (sub === subTarget ? " target" : "") }, ta,
    h("div", { class: "meta" }, "@", num("start"), "+", num("dur")),
    h("div", { class: "acts" },
      h("button", { class: "tiny", title: "Swap text with previous", disabled: j === 0, onclick: () => move(-1) }, "↑"),
      h("button", { class: "tiny", title: "Swap text with next", disabled: j === chunk.subs.length - 1, onclick: () => move(1) }, "↓"),
      h("button", { class: "tiny", title: "Insert subtitle after", onclick: () => mutate(() => {
        chunk.subs.splice(j + 1, 0, { text: "", start: sub.start + sub.dur, dur: 0 }); chunk.subsEdited = true; if (!chunk.timingEdited) redistribute(chunk);
      }) }, "+"),
      h("button", { class: "tiny", title: "Delete subtitle", onclick: () => mutate(() => { chunk.subs.splice(j, 1); if (!chunk.timingEdited) redistribute(chunk); }) }, "✕")),
    counts);
}

// ------------------------------------------------------------------ reference pane
function candidateList() {
  const out = [], seen = new Set();
  const add = (uk, extra) => { if (!uk || seen.has(uk) || !D.usa[uk] || !lineKeysOf(D.usa[uk]).length) return; seen.add(uk); out.push({ usa_key: uk, ...extra }); };
  for (const c of CAND.get(cur)?.candidates || []) add(c.usa_key, c);
  add(HUNG.get(cur)?.usa_key, { combined_score: HUNG.get(cur)?.score });
  add(cur, { sameId: true });
  add(conv.usa_ref, { manual: true });
  return out;
}

function renderRef() {
  const uk = conv.usa_ref;
  $("#usaKey").value = uk || "";
  const cands = candidateList(), top = cands[0]?.combined_score || 1;
  const box = $("#candidates"); box.replaceChildren();
  if (!cands.length) box.append(h("div", { class: "muted small" }, "No vector candidates (1-line bark?) — type a USA key above."));
  for (const c of cands) {
    const d = DUAL.get(c.usa_key);
    const mutual = d && d.line_best_jpn === cur && d.whole_best_jpn === cur;
    const oneSided = d && !mutual && (d.line_best_jpn === cur || d.whole_best_jpn === cur);
    const hung = HUNG.get(cur)?.usa_key === c.usa_key;
    const score = c.combined_score;
    box.append(h("div", { class: "cand" + (c.usa_key === uk ? " active" : ""), onclick: () => setRef(c.usa_key) },
      h("span", { class: "ck" }, c.usa_key),
      h("span", { class: "muted small" }, `${lineKeysOf(D.usa[c.usa_key]).length}L` + (c.line_f1 != null ? ` · F1 ${c.line_f1} · cos ${c.whole_cosine}` : "")),
      h("span", { class: "score" }, score != null ? score.toFixed(3) : "—"),
      h("div", { class: "bar" }, h("i", { style: `width:${score != null ? Math.max(0, (score / top) * 100) : 0}%` })),
      h("div", { class: "tags" },
        mutual && h("span", { class: "tag good", title: "USA→JPN line-F1 and whole-conv cosine both pick this JPN conv" }, "mutual best"),
        oneSided && h("span", { class: "tag warn", title: "Only one USA→JPN method picks this JPN conv" }, "one method agrees"),
        hung && h("span", { class: "tag good", title: "Global 1:1 Hungarian assignment (min 2 lines)" }, "hungarian 1:1"),
        c.sameId && h("span", { class: "tag" }, "same ID"),
        c.manual && h("span", { class: "tag" }, "manual pick"))));
  }

  const ol = $("#usaLines"); ol.replaceChildren();
  $("#usaTitle").textContent = uk ? uk : "";
  if (!uk || !D.usa[uk]) { $("#usaMeta").textContent = ""; ol.append(h("li", { class: "placeholder" }, "No reference selected.")); return; }
  const entry = D.usa[uk];
  $("#usaMeta").textContent = `${lineKeysOf(entry).length} lines`;
  for (const k of lineKeysOf(entry)) {
    const [s, d] = parseTiming(entry[1]?.[k]);
    ol.append(h("li", { "data-line": k },
      h("span", { class: "no" }, k), h("span", { class: "ut" }, usaText(entry[0][k])),
      h("button", { class: "tiny", title: "Replace the text of the target subtitle in the focused chunk (timing kept) · ⇧-click appends a new subtitle", onclick: (ev) => copyUsaLine(k, ev.shiftKey) }, "→"),
      h("span", { class: "tm" }, `${s} +${d}`, h("span", { class: "sim" }))));
  }
  highlightRef(true);
}

function setRef(uk) {
  if (!D.usa[uk]) { toast(`No USA entry ${uk}`, true); return; }
  if (conv.usa_ref === uk) return;
  pushUndo(); conv.usa_ref = uk; touch(); renderRef();
}

const norm = (s) => s.replace(/\s+/g, " ").trim();

// "use" on an engine suggestion: with several subtitles and one marked (clicked into), replace just its text;
// otherwise the subtitles become an auto-split of the suggestion over the chunk's JPN timing. Undoable.
function useMt(chunk, name, text) {
  mutate(() => {
    chunk.mt = text; chunk.decided = name; // decided locks the chunk against batch overwrites
    if (chunk.subs.length > 1 && chunk.subs.includes(subTarget)) { subTarget.text = wrapRows(text); chunk.subsEdited = true; }
    else { autoSubs(chunk, text); subTarget = chunk.subs[0]; }
  });
}

// → on a USA line: replace the text of the target subtitle (last one clicked into) in the focused chunk, keeping its
// timing; with no target, the chunk's subtitles become this one line over the chunk's JPN timing. ⇧-click appends instead.
function copyUsaLine(k, append = false) {
  const chunk = conv.chunks[focusIdx]; if (!chunk) return;
  const text = usaText(D.usa[conv.usa_ref][0][k]);
  const target = chunk.subs.includes(subTarget) ? subTarget : chunk.subs.length === 1 ? chunk.subs[0] : null;
  mutate(() => {
    if (append) {
      chunk.subs.push({ text, start: 0, dur: 0 });
      if (!chunk.timingEdited) redistribute(chunk);
    } else if (target) {
      target.text = text;
    } else {
      const [s, e] = chunkSpan(chunk);
      if (chunk.subs.length > 1) toast(`Replaced ${chunk.subs.length} subtitles with one (⌘Z to undo) — click into a subtitle first to replace just that one`);
      chunk.subs = [{ text, start: s, dur: Math.max(1, e - s) }];
      subTarget = chunk.subs[0];
    }
    chunk.subsEdited = true;
  });
}

// Highlight USA lines aligned (LaBSE) with the focused chunk; positional fallback otherwise.
function highlightRef(scroll) {
  const uk = conv?.usa_ref, chunk = conv?.chunks[focusIdx];
  const lis = [...document.querySelectorAll("#usaLines li[data-line]")];
  lis.forEach((li) => { li.classList.remove("hl", "hl2"); const s = li.querySelector(".sim"); if (s) s.textContent = ""; });
  if (!uk || !chunk || !lis.length) return;
  const al = D.alignments?.[cur]?.[uk];
  const best = new Map(); // usa line -> [sim, strong]
  if (al) {
    for (const jl of chunk.lines) for (const [ul, sim] of al[jl] || []) {
      const prev = best.get(ul)?.[0] ?? -1;
      if (sim > prev) best.set(ul, [sim, sim >= STRONG_SIM]);
    }
  } else {
    const jkeys = lineKeysOf(D.jpn[cur]);
    for (const jl of chunk.lines) {
      const idx = Math.round((jkeys.indexOf(jl) / Math.max(1, jkeys.length - 1)) * (lis.length - 1));
      best.set(lis[idx].dataset.line, [null, false]);
    }
  }
  let first = null;
  for (const li of lis) {
    const b = best.get(li.dataset.line); if (!b) continue;
    li.classList.add(b[1] ? "hl" : "hl2");
    li.querySelector(".sim").textContent = b[0] != null ? `  ${b[0].toFixed(2)}` : "  ~position";
    if (b[1] && !first) first = li;
  }
  first = first || lis.find((li) => li.classList.contains("hl2"));
  if (scroll && first) {
    const pane = $("#refPane"), r = first.getBoundingClientRect(), p = pane.getBoundingClientRect();
    if (r.top < p.top + 160 || r.bottom > p.bottom - 40) pane.scrollTo({ top: pane.scrollTop + r.top - p.top - p.height / 3, behavior: "smooth" });
  }
}

// ------------------------------------------------------------------ boot
function navigate(dir) {
  const i = listKeys.indexOf(cur);
  const next = listKeys[i < 0 ? 0 : i + dir];
  if (next) openConv(next);
}

function bind() {
  ["#search", "#sortBy", "#minLines", "#statusFilter", "#confFilter"].forEach((s) => $(s).addEventListener("input", renderList));
  $("#groupBtn").onclick = groupSelection;
  $("#ungroupBtn").onclick = () => ungroup();
  $("#translateAllBtn").onclick = translateAll;
  $("#undoBtn").onclick = undo; $("#redoBtn").onclick = redo;
  $("#statusSel").onchange = (e) => { conv.status = e.target.value; STATE.convs[cur] = conv; clearTimeout(saveTimer); save(cur, conv); renderList(); };
  $("#usaKey").onchange = (e) => { let v = e.target.value.trim(); if (/^\d+$/.test(v)) v = "vox-" + v.padStart(4, "0"); setRef(v); };
  const step = (dir) => { const keys = Object.keys(D.usa).filter((k) => lineKeysOf(D.usa[k]).length); const i = keys.indexOf(conv.usa_ref); setRef(keys[Math.max(0, Math.min(keys.length - 1, (i < 0 ? keys.indexOf(cur) : i) + dir))]); };
  $("#usaPrev").onclick = () => step(-1); $("#usaNext").onclick = () => step(1);
  $("#helpBtn").onclick = () => $("#help").showModal();
  $("#exportBtn").onclick = async () => {
    try {
      clearTimeout(saveTimer); if (conv) await save(cur, conv);
      const r = await api("/api/export", { filename: $("#exportName").value, line_break: $("#lineBreak").value === "\\r" ? "\r" : "｜", scope: $("#exportScope").value });
      toast(`Exported ${r.edited} edited conversations → ${r.path}`);
    } catch (e) { toast("Export failed: " + e.message, true, 7000); }
  };
  const trackShift = (ev) => (shiftHeld = ev.shiftKey);
  document.addEventListener("keyup", trackShift);
  window.addEventListener("blur", () => (shiftHeld = false));
  document.addEventListener("keydown", (ev) => {
    trackShift(ev);
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName);
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z" && !typing) { ev.preventDefault(); ev.shiftKey ? redo() : undo(); return; }
    if (typing || ev.metaKey || ev.ctrlKey || ev.altKey || !conv) return;
    const k = ev.key.toLowerCase();
    if (k === "g") groupSelection();
    else if (k === "u") ungroup();
    else if (k === "t") translateChunk(focusIdx).catch(() => {});
    else if (ev.key === "Escape") clearSel();
    else if (k === "[") navigate(-1);
    else if (k === "]") navigate(1);
    else if (k === "j" || ev.key === "ArrowDown") { ev.preventDefault(); setFocus(focusIdx + 1, true); }
    else if (k === "k" || ev.key === "ArrowUp") { ev.preventDefault(); setFocus(focusIdx - 1, true); }
    else return;
  });
  window.addEventListener("beforeunload", () => { if (conv && $("#saveState").textContent === "unsaved…") navigator.sendBeacon?.("/api/save", new Blob([JSON.stringify({ ds: DS, key: cur, conv })], { type: "application/json" })); });
}

(async function main() {
  D = await api("/api/data");
  for (const r of D.candidates) CAND.set(r.jpn_key, r);
  for (const r of D.dual) DUAL.set(r.usa_key, r);
  for (const r of D.hungarian) HUNG.set(r.jpn_key, r);
  STATE = D.state?.convs ? D.state : { convs: {} };
  $("#provider").textContent = `MT: ${D.engines.map((e) => e.name + (e.name === D.primary ? " (primary)" : "")).join(" + ")}`;
  $("#exportName").value = D.export_path;
  $("#lineBreak").value = D.line_break === "\r" ? "\\r" : "｜";
  const dsSel = $("#dataset");
  for (const d of D.datasets) dsSel.append(h("option", { value: d.name, selected: d.name === DS }, d.label));
  dsSel.onchange = () => { location.href = `/?ds=${encodeURIComponent(dsSel.value)}`; };
  $("#reviewLink").href = `review.html?ds=${encodeURIComponent(DS)}`;
  document.title = `${D.datasets.find((d) => d.name === DS)?.label || DS} · Undub Editor`;
  bind();
  renderList();
  const start = decodeURIComponent(location.hash.slice(1));
  openConv(D.jpn[start] ? start : listKeys[0]);
})().catch((e) => toast("Failed to load: " + e.message, true, 60000));
