"""Import an existing hand translation into a dataset's editor state, so editing resumes where it left off.

    .venv/bin/python vox_editor/import_translation.py --dataset vox \
        --from ../mgs1-undub/build-proprietary/vox/voxText-jpn-d1.json [--dry-run]

The translation file has the same [textDict, timingDict] format as the JPN source and may hold only
some of its entries. Every entry whose text differs from the JPN source (ignoring ‹TK› markup) counts as translated:
its conversation becomes status "done" with those subtitles. Each subtitle is attached to the last JPN
line that started before it (within SLACK frames), since translated lines are often split or retimed.
Chunks are flagged subsEdited/timingEdited so batch passes never overwrite them, and existing engine
output (mt/mts/mt_agree) is kept as alternates. Conversations already edited in the editor are skipped
unless --force. The state file is backed up first.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from server import ROOT, dataset, load_json, load_state, state_lock, state_path, write_state  # noqa: E402
from subtitles import parse_timing  # noqa: E402

SLACK = 5  # frames: retimed subtitles often start a frame or two before their JPN line
LOCKS = ("subsEdited", "timingEdited", "mtEdited", "decided")


def import_conv(key, texts, timings, src, old, same_id):
    jkeys = sorted(src[0])
    starts = [parse_timing(src[1].get(k))[0] for k in jkeys]
    alt = {tuple(c["lines"]): c for c in (old or {}).get("chunks", [])}
    chunks = []
    for k in jkeys:
        c = {"lines": [k], "mt": "", "subs": [], "subsEdited": True, "timingEdited": True}
        c.update({f: alt[(k,)][f] for f in ("mt", "mts", "mt_agree") if f in alt.get((k,), {})})
        chunks.append(c)
    for lk in sorted(texts):
        s, d = parse_timing(timings.get(lk))
        target = max([i for i, st in enumerate(starts) if st <= s + SLACK], default=0)
        chunks[target]["subs"].append({"text": re.sub(r"\r\n?|｜", "\n", texts[lk].strip()), "start": s, "dur": d})
    usa_ref = (old or {}).get("usa_ref") or (key if same_id else None)
    return {"status": "done", "usa_ref": usa_ref, "chunks": chunks, "imported": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name from vox_editor/datasets.json")
    ap.add_argument("--from", dest="src", required=True, help="existing translation JSON")
    ap.add_argument("--force", action="store_true", help="also replace conversations already edited in the editor")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = dataset(args.dataset)
    jpn = load_json(ROOT / d["jpn"])
    done = load_json(args.src)

    with state_lock(args.dataset):
        state = load_state(args.dataset)
        imported, unchanged, edited, unknown = [], [], [], []
        for key, (texts, timings) in sorted(done.items()):
            if key not in jpn:
                unknown.append(key)
                continue
            if {k: re.sub(r"‹[A-Z]+›", "", v) for k, v in texts.items()} == {k: re.sub(r"‹[A-Z]+›", "", v) for k, v in jpn[key][0].items()}:
                unchanged.append(key)
                continue
            old = state["convs"].get(key)
            if old and not args.force and any(c.get(f) for c in old.get("chunks", []) for f in LOCKS):
                edited.append(key)
                continue
            state["convs"][key] = import_conv(key, texts, timings, jpn[key], old, d.get("same_id"))
            imported.append(key)
        print(f"imported {len(imported)} conversations as done")
        print(f"left as-is: {len(unchanged)} identical to the JPN source {unchanged}, "
              f"{len(edited)} already edited in the editor {edited}, {len(unknown)} not in the JPN source {unknown}")
        if args.dry_run or not imported:
            return
        path = state_path(args.dataset)
        if path.exists():
            backup = path.with_name(f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
            shutil.copy2(path, backup)
            print(f"backup: {backup}")
        write_state(state, args.dataset)


if __name__ == "__main__":
    main()
