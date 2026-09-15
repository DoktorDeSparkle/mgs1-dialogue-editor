"""Inject vox-editor subtitles (voxRadio JSON from vox_to_radio.py) into an extracted RADIO.xml.

    python radio_inject.py voxRadio-d1.json RADIO.xml RADIO-vox.xml [--report report.json]

Standard library only, so it can live next to RadioDatTools/RadioDatRecompiler in mgs1-undub.
Then build as usual:  RadioDatRecompiler.py RADIO-vox.xml new-RADIO.DAT -s STAGE.DIR -S new-STAGE.DIR

For every VOX_CUES whose voxCode is in the JSON:
  * all SUBTITLE elements under it (document order, including ones inside IF/ELSE branches) are aligned to the
    conversation's JPN lines by text; IF/ELSE branches may repeat a line (one SUBTITLE per branch). The text
    must resemble those JPN lines - RADIO.DAT holds calls for both discs, so a voxCode alone can point at
    another disc's call. Otherwise the block is skipped.
  * each JPN line's SUBTITLE is replaced in place by that line's subtitles: text replaced, face/anim/unk3
    kept (speaker), extra subtitles inserted right after it (offset "<orig>-2", "-3"...), lines with no
    subtitles left removed. Neighbouring ANI_FACE / MUS_CUES / IF_CHECK stay where they are.
  * lines whose SUBTITLE sits inside a branch, or is repeated across branches, accept exactly one subtitle
    (every copy gets it); inserting/removing is only done for direct VOX_CUES children.
  * the Call gets modified="True". textHex is dropped on replaced subtitles (it no longer matches text).
Sizes are not touched: RadioDatRecompiler recomputes every container size from its children.
"""
import argparse
import difflib
import json
import re
import sys
import xml.etree.ElementTree as ET

JAPANESE = re.compile(r"[぀-ヿ㐀-鿿]")


def clean(s):
    s = re.sub(r"[#＃]｛([^｝、]*)[^｝]*｝[#＃]?", r"\1", s or "")
    s = re.sub(r"‹[A-Z]+›|\\r\\n|｜", "", s)
    return re.sub(r"[\s。、，．！？!?…・「」『』～〜ー―-]", "", s)


def sim(a, b):
    a, b = clean(a), clean(b)
    return 1.0 if not a and not b else difflib.SequenceMatcher(None, a, b).ratio()  # "……" vs "…。" compare equal


def align(radio_texts, jpn_texts):
    """Map each radio SUBTITLE (document order) to a vox line index. Positional when the counts match;
    otherwise a monotonic alignment where consecutive SUBTITLEs may share a line - IF/ELSE branches repeat
    a line once per branch (vox-0039: 12 SUBTITLEs for 9 lines). Returns (mapping, mean similarity) or (None, 0)."""
    n, m = len(radio_texts), len(jpn_texts)
    if n == m:
        return list(range(n)), sum(sim(a, b) for a, b in zip(radio_texts, jpn_texts)) / max(1, n)
    if n < m or m == 0:
        return None, 0.0
    s = [[sim(a, b) for b in jpn_texts] for a in radio_texts]
    NEG = float("-inf")
    best = [[NEG] * m for _ in range(n)]
    back = [[0] * m for _ in range(n)]
    best[0][0] = s[0][0]
    for i in range(1, n):
        for j in range(min(i + 1, m)):
            stay, step = best[i - 1][j], best[i - 1][j - 1] if j else NEG
            best[i][j], back[i][j] = (stay, j) if stay >= step else (step, j - 1)
            best[i][j] += s[i][j]
    mapping, j = [0] * n, m - 1
    for i in range(n - 1, -1, -1):
        mapping[i] = j
        j = back[i][j]
    return mapping, best[n - 1][m - 1] / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json")
    ap.add_argument("radio_in")
    ap.add_argument("radio_out")
    ap.add_argument("--min-sim", type=float, default=0.5, help="min JPN text similarity between radio block and vox lines")
    ap.add_argument("--report", help="write a JSON report here")
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    by_code = {int(v["voxCode"], 16): (k, v) for k, v in data.items()}
    root = ET.parse(args.radio_in).getroot()

    rep = {"updated": [], "skipped": [], "not_found": [], "japanese_left_in_modified_calls": []}
    found = set()
    for call in root.iter("Call"):
        parents = {child: p for p in call.iter() for child in p}
        touched = False
        for block in call.iter("VOX_CUES"):
            code = int(block.get("voxCode", "0"), 16)
            if code not in by_code:
                continue
            key, conv = by_code[code]
            found.add(key)
            where = f"{key} call {call.get('offset')} block {block.get('offset')}"
            subs = list(block.iter("SUBTITLE"))
            lines = sorted(conv["lines"])
            mapping, score = align([s.get("text") for s in subs], [conv["lines"][k] for k in lines])
            if mapping is None:
                rep["skipped"].append(f"{where}: {len(subs)} SUBTITLEs for {len(lines)} vox lines")
                continue
            if score < args.min_sim:
                rep["skipped"].append(f"{where}: JPN text similarity {score:.2f} (other disc's call, or already translated?)")
                continue
            pieces = {k: [s for s in conv["subtitles"] if s["line"] == k] for k in lines}
            targets = [lines[j] for j in mapping]
            fixed = {k for el, k in zip(subs, targets) if parents[el] is not block or targets.count(k) > 1}
            bad = sorted(k for k in fixed if len(pieces[k]) != 1)
            if bad:
                rep["skipped"].append(f"{where}: lines {bad} sit in IF/ELSE branches and need exactly one subtitle each")
                continue
            inserted = removed = 0
            for el, k in zip(subs, targets):
                parent = parents[el]
                idx = list(parent).index(el)
                parent.remove(el)
                if not pieces[k]:
                    removed += 1
                for n, piece in enumerate(pieces[k]):
                    if piece["text"] is None:
                        new = el  # untranslated: keep the original element
                    else:
                        new = ET.Element("SUBTITLE", {a: v for a, v in el.attrib.items() if a != "textHex"})
                        new.tail = el.tail  # keep the XML indentation
                        new.set("text", piece["text"].replace("\r\n", "\n").replace("\n", "\\r\\n"))
                        if n:
                            new.set("offset", f"{el.get('offset')}-{n + 1}")
                            inserted += 1
                    parent.insert(idx + n, new)
            touched = True
            rep["updated"].append(f"{where}: {len(lines)} lines -> {len(conv['subtitles'])} subtitles"
                                  + (f" ({len(subs)} SUBTITLEs incl. branch alternates)" if len(subs) != len(lines) else "")
                                  + (f" (+{inserted} inserted)" if inserted else "") + (f" (-{removed} removed)" if removed else ""))
        if touched:
            call.set("modified", "True")
            left = [s.get("text") for s in call.iter("SUBTITLE") if JAPANESE.search(s.get("text") or "")]
            if left:
                rep["japanese_left_in_modified_calls"].append(f"call {call.get('offset')}: {len(left)} JPN subtitles, e.g. {left[0][:30]}")

    rep["not_found"] = sorted(set(data) - found)
    ET.ElementTree(root).write(args.radio_out, encoding="utf-8", xml_declaration=True)

    print(f"updated {len(rep['updated'])} VOX_CUES blocks for {len({u.split()[0] for u in rep['updated']})} conversations; "
          f"skipped {len(rep['skipped'])} blocks; {len(rep['not_found'])} conversations have no VOX_CUES in this RADIO.xml")
    for s in rep["skipped"]:
        print("  skipped:", s)
    if rep["japanese_left_in_modified_calls"]:
        print(f"  {len(rep['japanese_left_in_modified_calls'])} modified calls still contain Japanese subtitles "
              "(recompiler drops graphicsBytes for modified calls - check kanji rendering)")
    if args.report:
        json.dump(rep, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
