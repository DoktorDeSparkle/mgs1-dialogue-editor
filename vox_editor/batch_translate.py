"""Full machine-translation pass over a dataset (vox, demo, zMovie; see datasets.json) into its editor state file.

    .venv/bin/python vox_editor/batch_translate.py                 # all engines in config.json
    .venv/bin/python vox_editor/batch_translate.py --engines sugoi-14b-ultra --only vox-0007 vox-0012
    .venv/bin/python vox_editor/batch_translate.py --score         # LaBSE agreement between engines (after translating)
    .venv/bin/python vox_editor/batch_translate.py --dataset demo-d1   # any dataset in datasets.json (default: vox)

Resumable: chunks that already have an engine's output are skipped. Safe to run
while the editor/review tool is open (shared file lock); chunks you have
edited or decided only get alternates filled in, never overwritten.

Per chunk it stores  mts: {engine: text}; the primary engine's text also becomes
`mt` plus auto-split subtitles (same rules as the editor UI). New conversations
get status "mt" (MT draft). Chunks you edited or decided keep their chosen text.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from server import ROOT, dataset, load_config, load_json, load_state, state_lock, subtitle_rows, translate, write_state  # noqa: E402
from subtitles import auto_subs, jpn_plain, needs_mt  # noqa: E402


# ---------------------------------------------------------------- state

def new_conv(entry):
    return {"status": "mt", "usa_ref": None, "chunks": [{"lines": [k], "mt": "", "subs": []} for k in sorted(entry[0])]}


def chunk_locked(chunk):
    """Chunks the user has edited or decided in the review tool keep their chosen text."""
    return any(chunk.get(f) for f in ("subsEdited", "timingEdited", "mtEdited", "decided"))


def chunk_text(chunk, entry):
    return "".join(jpn_plain(entry[0][k]) for k in chunk["lines"] if k in entry[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vox", help="dataset name from vox_editor/datasets.json")
    ap.add_argument("--engines", nargs="*", help="engine names from config.json (default: all)")
    ap.add_argument("--only", nargs="*", help="only these vox keys")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests per engine (match LM Studio's parallel slots)")
    ap.add_argument("--score", action="store_true", help="compute LaBSE agreement between engines instead of translating")
    args = ap.parse_args()
    ds = args.dataset
    if args.score:
        return score(ds)

    cfg = load_config()
    engines = args.engines or list(cfg["engines"])
    primary = cfg["primary"]
    dcfg = dataset(ds)
    jpn = load_json(ROOT / dcfg["jpn"])
    usa = load_json(ROOT / dcfg["usa"])
    cands = {r["jpn_key"]: r for r in load_json(ROOT / dcfg["candidates"], [])} if dcfg.get("candidates") else {}
    hung = {r["jpn_key"]: r for r in load_json(ROOT / dcfg["hungarian"], [])} if dcfg.get("hungarian") else {}

    # Seed conversations and collect jobs from a snapshot of the state.
    with state_lock(ds):
        state = load_state(ds)
        for key, entry in jpn.items():
            if args.only and key not in args.only:
                continue
            if not any(needs_mt(jpn_plain(t)) for t in entry[0].values()):
                continue
            if key not in state["convs"]:
                conv = new_conv(entry)
                same = key if dcfg.get("same_id") and (usa.get(key) or [{}])[0] else None
                conv["usa_ref"] = same or ((cands.get(key) or {}).get("candidates") or [{}])[0].get("usa_key") or (hung.get(key) or {}).get("usa_key")
                state["convs"][key] = conv
        write_state(state, ds)

    jobs = []  # (engine, key, lines-tuple, text, context)
    for key, conv in state["convs"].items():
        if args.only and key not in args.only or key not in jpn:
            continue
        entry = jpn[key]
        texts = [chunk_text(c, entry) for c in conv["chunks"]]
        for i, chunk in enumerate(conv["chunks"]):
            if not needs_mt(texts[i]):
                continue
            context = "\n".join(("▶ " if j == i else "  ") + t for j, t in enumerate(texts))
            for eng in engines:
                if eng not in (chunk.get("mts") or {}):
                    jobs.append((eng, key, tuple(chunk["lines"]), texts[i], context))

    total = len(jobs)
    print(f"[{ds}] {total} translations to run ({', '.join(engines)}), {args.workers} parallel per engine", flush=True)
    if not total:
        return

    pending, lock = [], threading.Lock()
    done = {e: 0 for e in engines}
    errors = []
    t0 = time.time()

    def flush():
        with lock:
            batch = pending[:]
            pending.clear()
        if not batch:
            return
        with state_lock(ds):
            st = load_state(ds)
            for eng, key, lines, text in batch:
                conv = st["convs"].get(key)
                if not conv:
                    continue
                chunk = next((c for c in conv["chunks"] if tuple(c["lines"]) == lines), None)
                if chunk is None:  # regrouped in the editor meanwhile; drop, a rerun will pick it up
                    continue
                chunk.setdefault("mts", {})[eng] = text
                touched = chunk_locked(chunk) or conv.get("status") == "done"
                if eng == primary and not touched:
                    chunk["mt"] = text
                    auto_subs(chunk, text, jpn[key][1], subtitle_rows(ds, key))
                elif not chunk.get("mt") and not touched and primary not in engines:
                    chunk["mt"] = text  # primary not part of this run: fall back to whatever we have
                    auto_subs(chunk, text, jpn[key][1], subtitle_rows(ds, key))
            write_state(st, ds)

    def run(job):
        eng, key, lines, text, context = job
        for attempt in range(3):
            try:
                out = translate({"text": text, "context": context, "engine": eng})
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    errors.append((eng, key, lines, f"{type(e).__name__}: {e}"))
                    return eng
                time.sleep(3 * (attempt + 1))
        with lock:
            pending.append((eng, key, lines, out))
        return eng

    pools = {e: ThreadPoolExecutor(args.workers) for e in engines}
    futures = [pools[j[0]].submit(run, j) for j in jobs]
    last_flush = time.time()
    for n, fut in enumerate(as_completed(futures), 1):
        done[fut.result()] += 1
        if time.time() - last_flush > 10 or n == total:
            flush()
            last_flush = time.time()
            rate = n / (time.time() - t0)
            eta = (total - n) / rate if rate else 0
            per = "  ".join(f"{e} {done[e]}" for e in engines)
            print(f"[{n}/{total}] {per}  {rate:.1f}/s  eta {eta / 60:.0f} min  errors {len(errors)}", flush=True)
    for p in pools.values():
        p.shutdown()
    flush()
    if errors:
        (ROOT / f"batch_translate_errors-{ds}.json").write_text(json.dumps(errors, ensure_ascii=False, indent=1))
        print(f"{len(errors)} failures written to batch_translate_errors-{ds}.json — rerun to retry them", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- agreement scoring

def score(ds="vox"):
    """LaBSE cosine between engines' English outputs per chunk -> chunk.mt_agree,
    conv.mt_agree = min over chunks. Low agreement = the models read the line
    differently = review it first."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    with state_lock(ds):
        state = load_state(ds)
    items = []  # (key, lines, [texts])
    for key, conv in state["convs"].items():
        for c in conv["chunks"]:
            mts = [t for t in (c.get("mts") or {}).values() if t]
            if len(mts) >= 2:
                items.append((key, tuple(c["lines"]), mts))
    if not items:
        print("no chunks with 2+ engine outputs yet")
        return
    model = SentenceTransformer("sentence-transformers/LaBSE")
    flat = [t for _, _, mts in items for t in mts]
    embs = model.encode(flat, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    scores, i = {}, 0
    for key, lines, mts in items:
        e = embs[i:i + len(mts)]
        i += len(mts)
        sim = e @ e.T
        scores[(key, lines)] = round(float(sim[np.triu_indices(len(mts), 1)].min()), 3)

    with state_lock(ds):
        st = load_state(ds)
        for key, conv in st["convs"].items():
            vals = []
            for c in conv["chunks"]:
                s = scores.get((key, tuple(c["lines"])))
                if s is not None:
                    c["mt_agree"] = s
                    vals.append(s)
            if vals:
                conv["mt_agree"] = min(vals)
        write_state(st, ds)
    allv = np.array(list(scores.values()))
    print(f"scored {len(allv)} chunks: p10 {np.percentile(allv, 10):.3f}  p25 {np.percentile(allv, 25):.3f}  median {np.median(allv):.3f}")


if __name__ == "__main__":
    main()
