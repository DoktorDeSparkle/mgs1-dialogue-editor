"""Vox undub subtitle editor — local web GUI.

Run:  .venv/bin/python vox_editor/server.py   (then open http://127.0.0.1:8765)

Stdlib only (no Flask). Serves the static UI, the vox data + vector-match
metadata, persists editing state to vox_editor_state.json, proxies translation
requests (so API keys stay server-side and CORS is a non-issue), and exports a
voxText-jpn-format JSON with the edited English subtitles.

Translation provider is configured via env vars or vox_editor/config.json
(see config.example.json).
"""
import contextlib
import copy
import fcntl
import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
STATE_PATH = Path(os.environ.get("VOX_EDITOR_STATE", ROOT / "vox_editor_state.json"))
DEFAULT_EXPORT = ROOT / "voxText-jpn-en.json"
PORT = int(os.environ.get("VOX_EDITOR_PORT", "8765"))


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def load_config():
    """Returns {"primary": name, "engines": {name: engine_cfg}}.

    config.json either holds one engine at top level (legacy) or
    {"primary": "...", "engines": {...}}. The primary engine's output becomes the
    draft subtitles; the others are kept alongside as alternates."""
    cfg = load_json(Path(__file__).parent / "config.json", {}) or {}
    if "engines" not in cfg:
        cfg = {"primary": "default", "engines": {"default": cfg}}
    env = {
        "provider": os.environ.get("VOX_TRANSLATE_PROVIDER"),
        "url": os.environ.get("VOX_TRANSLATE_URL"),
        "model": os.environ.get("VOX_TRANSLATE_MODEL"),
        "api_key": os.environ.get("VOX_TRANSLATE_API_KEY"),
    }
    if any(env.values()):  # env vars override the primary engine
        cfg["engines"][cfg["primary"]] = {**cfg["engines"].get(cfg["primary"], {}), **{k: v for k, v in env.items() if v}}
    for e in cfg["engines"].values():
        e.setdefault("provider", "openai")
    cfg.setdefault("primary", next(iter(cfg["engines"])))
    return cfg


@contextlib.contextmanager
def state_lock():
    """Cross-process lock so the editor and batch_translate.py never clobber each other's writes."""
    with open(STATE_PATH.with_suffix(".lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def write_state(state):
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


# ---------------------------------------------------------------- translation

SYSTEM_PROMPT = (
    "You translate Japanese dialogue from the video game Metal Gear Solid (1998) into "
    "natural English subtitles. Keep speaker tone, keep it concise enough to read as a "
    "subtitle, and use established series terminology (Snake, Colonel, FOXHOUND, Codec, "
    "Shadow Moses, etc). Output ONLY the translation, no notes or quotes."
)


def trim_context(context, n):
    """Keep n lines either side of the "▶"-marked line (Sugoi degrades with long context)."""
    lines = context.split("\n")
    i = next((j for j, l in enumerate(lines) if l.startswith("▶")), 0)
    return "\n".join(lines[max(0, i - n): i + n + 1])


def build_user_prompt(req, engine=None):
    engine = engine or {}
    parts = []
    context = req.get("context")
    if context and engine.get("context_lines"):
        context = trim_context(context, engine["context_lines"])
    if context:
        parts.append("Surrounding conversation (Japanese, for context only):\n" + context)
    if req.get("reference") and engine.get("use_reference", True):
        parts.append(
            "Official US localization of what is probably the same conversation "
            "(may be loosely adapted; use for terminology/names only):\n" + req["reference"]
        )
    parts.append("Translate this line:\n" + req["text"])
    return "\n\n".join(parts)


def http_json(url, payload, headers):
    data = json.dumps(payload).encode("utf-8")
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(r, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def translate(req, cfg=None):
    conf = load_config()
    name = req.get("engine") or conf["primary"]
    cfg = cfg or conf["engines"].get(name)
    if cfg is None:
        raise ValueError(f"unknown engine {name!r}; configured: {', '.join(conf['engines'])}")
    provider = cfg["provider"]
    text = req["text"]
    system = cfg.get("system_prompt", SYSTEM_PROMPT)
    user = build_user_prompt(req, cfg)

    if provider == "openai":  # any OpenAI-compatible endpoint: OpenAI, LM Studio, Ollama, vLLM...
        url = cfg.get("url") or "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
        out = http_json(url, {
            "model": cfg.get("model", "gpt-4o-mini"),
            "temperature": cfg.get("temperature", 0.3),
            **cfg.get("extra", {}),  # e.g. {"reasoning_effort": "none"} to disable Qwen thinking in LM Studio
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }, headers)
        text = out["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    if provider == "anthropic":
        url = cfg.get("url") or "https://api.anthropic.com/v1/messages"
        out = http_json(url, {
            "model": cfg.get("model", "claude-sonnet-5"),
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }, {"x-api-key": cfg.get("api_key", ""), "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out["content"]).strip()

    if provider == "deepl":
        url = cfg.get("url") or "https://api-free.deepl.com/v2/translate"
        payload = {"text": [text], "source_lang": "JA", "target_lang": "EN-US"}
        if req.get("context"):
            payload["context"] = req["context"]
        out = http_json(url, payload, {"Authorization": f"DeepL-Auth-Key {cfg.get('api_key', '')}"})
        return out["translations"][0]["text"].strip()

    if provider == "libretranslate":
        url = cfg.get("url") or "http://127.0.0.1:5000/translate"
        payload = {"q": text, "source": "ja", "target": "en", "format": "text"}
        if cfg.get("api_key"):
            payload["api_key"] = cfg["api_key"]
        return http_json(url, payload, {})["translatedText"].strip()

    if provider == "mock":  # for UI testing without an endpoint
        return f"[MT] {text}"

    raise ValueError(f"unknown translation provider: {provider}")


# ---------------------------------------------------------------- export

def fmt_timing(start, dur):
    return f"{int(round(start))},{int(round(dur))}"


def export(state, jpn, path, line_break):
    """Write a voxText-jpn-format file. Convs with saved edits get their English
    subtitles (renumbered 01..NN); untouched convs keep the original JPN entry.
    Chunks with no subtitles yet fall back to their original JPN lines, so a
    half-finished conversation never silently loses lines."""
    out, edited = {}, 0
    for key, orig in jpn.items():
        conv = state.get("convs", {}).get(key)
        chunks = (conv or {}).get("chunks") or []
        if not any(s.get("text", "").strip() for c in chunks for s in c.get("subs", [])):
            out[key] = orig
            continue
        rows = []  # (text, timing)
        for chunk in chunks:
            subs = [s for s in chunk.get("subs", []) if s.get("text", "").strip()]
            if subs:
                rows += [(s["text"].strip().replace("\n", line_break), fmt_timing(s["start"], s["dur"])) for s in subs]
            else:
                rows += [(orig[0][k], orig[1].get(k, "0,0")) for k in chunk["lines"] if k in orig[0]]
        out[key] = [{f"{i:02d}": t for i, (t, _) in enumerate(rows, 1)},
                    {f"{i:02d}": tm for i, (_, tm) in enumerate(rows, 1)}]
        edited += 1
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")
    return edited


def merge_background(disk, incoming):
    """The editor page may hold a stale copy of a conversation while batch_translate.py
    or the review page write to disk. Keep engine outputs / scores / decisions that
    landed on disk after the page loaded, for chunks the page didn't touch."""
    if not disk:
        return incoming
    by_lines = {tuple(c["lines"]): c for c in disk.get("chunks", [])}
    for c in incoming.get("chunks", []):
        d = by_lines.get(tuple(c["lines"]))
        if not d:
            continue
        c["mts"] = {**(d.get("mts") or {}), **(c.get("mts") or {})}
        for f in ("mt_agree", "decided"):
            if f in d and f not in c:
                c[f] = d[f]
        page_touched = any(c.get(f) for f in ("subsEdited", "timingEdited", "mtEdited"))
        if not page_touched and (d.get("decided") or (d.get("mt") and not c.get("mt"))):
            c["mt"], c["subs"] = d.get("mt", ""), d.get("subs", [])
    for f in ("mt_agree",):
        if f in disk and f not in incoming:
            incoming[f] = disk[f]
    return incoming


# ---------------------------------------------------------------- review decisions

_JPN = None


def decide(req):
    """Apply an A/B review decision to the *current* state on disk (the batch pass may
    be writing concurrently, so never save a whole stale conversation).
    req: {key, lines, choice: engine name | "edit", text} or {key, lines, restore: chunk}."""
    global _JPN
    from subtitles import auto_subs

    if _JPN is None:
        _JPN = load_json(ROOT / "voxText-jpn.json")
    with state_lock():
        state = load_json(STATE_PATH, {"convs": {}})
        conv = state["convs"].get(req["key"])
        chunk = next((c for c in (conv or {}).get("chunks", []) if c["lines"] == req["lines"]), None)
        if chunk is None:
            raise ValueError(f"{req['key']} lines {req['lines']} no longer exist — regrouped in the editor? Reload the queue.")
        prev = copy.deepcopy(chunk)
        if "restore" in req:
            chunk.clear()
            chunk.update(req["restore"])
        else:
            chunk["mt"] = req["text"]
            chunk["decided"] = req["choice"]
            if req["choice"] == "edit":
                chunk["mtEdited"] = True
            if not chunk.get("subsEdited") and not chunk.get("timingEdited"):
                auto_subs(chunk, req["text"], _JPN[req["key"]][1])
        conv["updated"] = __import__("datetime").datetime.now().isoformat()
        write_state(state)
    return {"ok": True, "prev": prev, "chunk": chunk}


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            sys.stderr.write("%s\n" % (fmt % args))

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        if self.path == "/api/data":
            cfg = load_config()
            return self.send_json({
                "usa": load_json(ROOT / "voxText-usa.json"),
                "jpn": load_json(ROOT / "voxText-jpn.json"),
                "candidates": load_json(ROOT / "jpn_to_usa_candidates.json", []),
                "dual": load_json(ROOT / "vox_dual_scores.json", []),
                "hungarian": load_json(ROOT / "vox_matches_min2.json", []),
                "alignments": load_json(ROOT / "line_alignments.json", {}),
                "state": load_json(STATE_PATH, {"convs": {}}),
                "primary": cfg["primary"],
                "engines": [{"name": n, "provider": e["provider"], "model": e.get("model")} for n, e in cfg["engines"].items()],
                "export_path": str(DEFAULT_EXPORT.name),
            })
        path = "/index.html" if self.path in ("/", "") else self.path.split("?")[0]
        f = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in f.parents or not f.is_file():
            self.send_error(404)
            return
        ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}.get(f.suffix, "application/octet-stream")
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            req = self.read_body()
            if self.path == "/api/save":
                with state_lock():
                    state = load_json(STATE_PATH, {"convs": {}})
                    state["convs"][req["key"]] = merge_background(state["convs"].get(req["key"]), req["conv"])
                    write_state(state)
                return self.send_json({"ok": True})
            if self.path == "/api/decide":
                return self.send_json(decide(req))
            if self.path == "/api/translate":
                return self.send_json({"text": translate(req), "engine": req.get("engine") or load_config()["primary"]})
            if self.path == "/api/export":
                name = Path(req.get("filename") or DEFAULT_EXPORT.name).name  # keep inside project dir
                if not name.endswith(".json") or name in ("voxText-jpn.json", "voxText-usa.json"):
                    raise ValueError("refusing to overwrite source data; pick another .json name")
                with state_lock():
                    state = load_json(STATE_PATH, {"convs": {}})
                n = export(state, load_json(ROOT / "voxText-jpn.json"), ROOT / name, req.get("line_break", "\r"))
                return self.send_json({"ok": True, "path": name, "edited": n})
            self.send_error(404)
        except urllib.error.HTTPError as e:
            self.send_json({"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"}, 502)
        except Exception as e:  # surface everything to the UI
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Vox editor on http://127.0.0.1:{PORT}  (engines: {', '.join(cfg['engines'])}; primary: {cfg['primary']})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
