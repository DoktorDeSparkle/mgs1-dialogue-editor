"""Import radio story-call translations (storyCalls-v2.json) into the vox editor state.

    .venv/bin/python vox_editor/import_story_calls.py [--dry-run]

storyCalls-v2.json is keyed call offset -> VOX_CUES offset -> SUBTITLE offset -> English text, all offsets
in the JPN disc 1 RADIO.xml. Each VOX_CUES block names its voice clip (voxCode = clip byte offset / 0x800
in VOX.DAT), so the block maps to a vox conversation, and its i-th SUBTITLE is that conversation's i-th
line (the game pairs radio text with vox timings by position). Blocks whose SUBTITLE count doesn't match
the vox line count are skipped.

Imported lines overwrite the editor state (conversation becomes "done", chunks locked). A line whose
existing subtitles already carry the same text (e.g. split or retimed) keeps them. New subtitles get the
JPN line's timing. The state file is backed up first.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from server import ROOT, load_json, load_state, state_lock, state_path, write_state  # noqa: E402
from subtitles import parse_timing  # noqa: E402

UNDUB = ROOT.parent / "mgs1-undub"
STORY = UNDUB / "build-proprietary/radio/storyCalls-v2.json"
RADIO = UNDUB / "workingFiles/jpn-d1/radio/RADIO.xml"
OFFSETS = UNDUB / "workingFiles/jpn-d1/vox/newVoxOffsets.json"

norm = lambda s: " ".join(s.split())  # noqa: E731


def story_lines(story, radio, code2vox, jpn):
    """-> ({vox_key: {line_key: text}}, report dict)"""
    blocks = {}
    for m in re.finditer(r'<VOX_CUES offset="(\d+)"[^>]*voxCode="([0-9a-f]+)"[^>]*>(.*?)</VOX_CUES>', radio, re.S):
        blocks[m.group(1)] = (int(m.group(2), 16), re.findall(r'<SUBTITLE offset="(\d+)"', m.group(3)))
    out, rep = {}, {"skipped_blocks": [], "conflicts": []}
    for call, voxes in story.items():
        for vox_off, subs in voxes.items():
            if vox_off not in blocks or blocks[vox_off][0] == 0:
                rep["skipped_blocks"].append(f"call {call} block {vox_off}: not a voiced VOX_CUES ({len(subs)} subs)")
                continue
            code, sub_offs = blocks[vox_off]
            key = code2vox[code]
            lines = sorted(jpn[key][0])
            if len(sub_offs) != len(lines):
                rep["skipped_blocks"].append(f"call {call} block {vox_off} ({key}): {len(sub_offs)} subtitles vs {len(lines)} vox lines")
                continue
            for so, line in zip(sub_offs, lines):
                if so not in subs:
                    continue
                text = subs[so].replace("\\r\\n", "\n").strip()
                prev = out.setdefault(key, {}).get(line)
                if prev is None:
                    out[key][line] = text
                elif norm(prev) != norm(text):
                    rep["conflicts"].append(f"{key} line {line}: kept {prev!r}, other copy {text!r}")
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jpn = load_json(ROOT / "voxText-jpn.json")
    code2vox = {int(v, 16) // 0x800: f"vox-{k}" for k, v in load_json(OFFSETS).items()}
    lines_by_vox, rep = story_lines(load_json(STORY), RADIO.read_text(encoding="utf-8"), code2vox, jpn)

    with state_lock("vox"):
        state = load_state("vox")
        kept = replaced = 0
        touched_tonight = []
        for key, texts in sorted(lines_by_vox.items()):
            old = state["convs"].get(key) or {}
            if old.get("updated") and not old.get("imported"):
                touched_tonight.append(key)
            old_by_line = {}
            for c in old.get("chunks", []):
                for k in c["lines"]:
                    old_by_line[k] = c
            chunks = []
            for k in sorted(jpn[key][0]):
                oc = old_by_line.get(k)
                single = oc if oc and oc["lines"] == [k] else None
                c = {"lines": [k], "mt": "", "subs": [], "subsEdited": True, "timingEdited": True}
                if single:
                    c.update({f: single[f] for f in ("mt", "mts", "mt_agree") if f in single})
                if k not in texts:  # no story text for this line: keep what was there
                    c["subs"] = single["subs"] if single else []
                elif single and norm(" ".join(s["text"] for s in single["subs"])) == norm(texts[k]):
                    c["subs"] = single["subs"]
                    kept += 1
                else:
                    s, d = parse_timing(jpn[key][1].get(k))
                    c["subs"] = [{"text": texts[k], "start": s, "dur": d}]
                    replaced += 1
                chunks.append(c)
            state["convs"][key] = {"status": "done", "usa_ref": old.get("usa_ref"), "chunks": chunks,
                                   "imported": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "storyCalls-v2"}

        print(f"{len(lines_by_vox)} vox conversations from story calls: {replaced} lines written, {kept} kept (same text already)")
        print(f"overwrote conversations edited in the editor: {touched_tonight}")
        for s in rep["skipped_blocks"]:
            print("skipped:", s)
        for s in rep["conflicts"]:
            print("conflict:", s)
        if args.dry_run:
            return
        path = state_path("vox")
        backup = path.with_name(f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        shutil.copy2(path, backup)
        print(f"backup: {backup}")
        write_state(state, "vox")


if __name__ == "__main__":
    main()
