"""Export edited vox conversations for the game build: one JSON that drives both RADIO (text) and VOX (timings).

    .venv/bin/python vox_editor/vox_to_radio.py [--scope reviewed|done|all] [--only vox-0078 ...]

Writes
  exports/voxRadio-d1.json          -> radio_inject.py (RADIO.xml subtitles)
  exports/voxText-jpn-undub-d1.json -> mgs1-undub voxTextInjector.py (VOX timings; same subtitles, same order)

voxRadio JSON, per conversation:
  "vox-0078": {
    "voxCode": "000041be",               # clip byte offset in VOX.DAT / 0x800, as in RADIO.xml VOX_CUES
    "lines": {"01": "<JPN text>", ...},  # original JPN lines, so the injector can verify it found the right block
    "subtitles": [                       # time order = the order the game pairs radio text with vox timings
      {"line": "03", "start": 86, "dur": 35, "text": "To make things worse..."},
      {"line": "03", "start": 121, "dur": 35, "text": "...high-voltage current in the floor."},   # inserted
      {"line": "04", "start": 168, "dur": 83, "text": null},   # null = no English yet, keep the JPN radio line
    ]}
Each subtitle's "line" is the JPN line it belongs to; its SUBTITLE element supplies the speaker (face/anim).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from server import EXPORT_DIR, EXPORT_SCOPES, ROOT, load_json, load_state  # noqa: E402
from subtitles import parse_timing  # noqa: E402

OFFSETS = ROOT.parent / "mgs1-undub/workingFiles/jpn-d1/vox/newVoxOffsets.json"
SLACK = 5  # frames


def conv_subtitles(conv, entry):
    """Ordered subtitles for one conversation, each tagged with its JPN line (None text = keep JPN)."""
    texts, timings = entry
    starts = {k: parse_timing(timings.get(k))[0] for k in texts}
    rows = []  # (start, chunk index, sub index, row)
    for ci, chunk in enumerate(conv.get("chunks", [])):
        lines = [k for k in chunk["lines"] if k in texts]
        subs = [s for s in chunk.get("subs", []) if s.get("text", "").strip()]
        if not subs:  # untranslated chunk: its JPN lines pass through unchanged
            for li, k in enumerate(lines):
                s, d = parse_timing(timings.get(k))
                rows.append((s, ci, li, {"line": k, "start": s, "dur": d, "text": None}))
            continue
        for si, sub in enumerate(subs):
            before = [k for k in lines if starts[k] <= sub["start"] + SLACK]
            line = before[-1] if before else lines[0]
            rows.append((sub["start"], ci, si, {"line": line, "start": int(sub["start"]), "dur": int(sub["dur"]),
                                                "text": sub["text"].strip()}))
    rows.sort(key=lambda r: r[:3])
    return [r[3] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="reviewed", choices=list(EXPORT_SCOPES), help="reviewed = done + in progress")
    ap.add_argument("--only", nargs="*", help="only these vox keys")
    ap.add_argument("--out", default=str(EXPORT_DIR / "voxRadio-d1.json"))
    ap.add_argument("--vox-out", default=str(EXPORT_DIR / "voxText-jpn-undub-d1.json"))
    args = ap.parse_args()

    jpn = load_json(ROOT / "voxText-jpn.json")
    offsets = load_json(OFFSETS)
    state = load_state("vox")
    statuses = EXPORT_SCOPES[args.scope]

    radio, vox = {}, {}
    for key, conv in sorted(state["convs"].items()):
        if key not in jpn or (args.only and key not in args.only):
            continue
        if statuses is not None and conv.get("status") not in statuses:
            continue
        subs = conv_subtitles(conv, jpn[key])
        if not any(s["text"] for s in subs):
            continue
        radio[key] = {"voxCode": f"{int(offsets[key[4:]], 16) // 0x800:08x}", "lines": jpn[key][0], "subtitles": subs}
        vox[key] = [{f"{i:02d}": (s["text"].replace("\n", "｜") if s["text"] else jpn[key][0][s["line"]]) for i, s in enumerate(subs, 1)},
                    {f"{i:02d}": f"{s['start']},{s['dur']}" for i, s in enumerate(subs, 1)}]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(radio, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.vox_out).write_text(json.dumps(vox, ensure_ascii=False, indent=4), encoding="utf-8")
    n = sum(len(v["subtitles"]) for v in radio.values())
    changed = sum(1 for k, v in radio.items() if len(v["subtitles"]) != len(v["lines"]))
    print(f"{len(radio)} conversations ({args.scope}), {n} subtitles; {changed} with a different subtitle count than JPN lines")
    print(f"-> {args.out}\n-> {args.vox_out}")


if __name__ == "__main__":
    main()
