"""Precompute per-line JPN -> USA alignments for each JPN conv's top candidates.

Used by the vox editor GUI to highlight which USA reference lines correspond to
a selected JPN line/chunk. Reads jpn_to_usa_candidates.json (run
jpn_to_usa_table.py first).

Output: line_alignments.json
  { jpn_key: { usa_key: { jpn_line_no: [[usa_line_no, sim], ...top 2] } } }
Line numbers are the original "01".."NN" dict keys (empty lines skipped).
"""
import json
import re
import sys

from sentence_transformers import SentenceTransformer

TOP_N = 2

CLEAN_PATTERNS = [
    (r"‹[A-Z]+›", ""),
    (r"[｜\r\n]", " "),
    (r"＃｛([^｝]*)｝‹?T?K?›?＃?", lambda m: m.group(1).split("、")[0]),
    (r"＃|｛|｝", ""),
]


def clean(text):
    for pat, repl in CLEAN_PATTERNS:
        text = re.sub(pat, repl, text)
    return text.strip()


def load_lines(path):
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for key, val in raw.items():
        if not val or not val[0]:
            continue
        pairs = [(k, clean(v)) for k, v in sorted(val[0].items())]
        pairs = [(k, t) for k, t in pairs if t]
        if pairs:
            out[key] = pairs
    return out


def main():
    usa = load_lines("voxText-usa.json")
    jpn = load_lines("voxText-jpn.json")
    cands = json.load(open("jpn_to_usa_candidates.json", encoding="utf-8"))

    model = SentenceTransformer("sentence-transformers/LaBSE")
    cache = {}

    def embed(side, key, lines):
        if (side, key) not in cache:
            cache[(side, key)] = model.encode([t for _, t in lines], normalize_embeddings=True)
        return cache[(side, key)]

    result = {}
    for i, row in enumerate(cands):
        jk = row["jpn_key"]
        if jk not in jpn or not row["candidates"]:
            continue
        je = embed("jpn", jk, jpn[jk])
        per_usa = {}
        for c in row["candidates"]:
            uk = c["usa_key"]
            if uk not in usa:
                continue
            ue = embed("usa", uk, usa[uk])
            sim = je @ ue.T
            per_line = {}
            for li, (jline, _) in enumerate(jpn[jk]):
                order = sim[li].argsort()[::-1][:TOP_N]
                per_line[jline] = [[usa[uk][o][0], round(float(sim[li, o]), 3)] for o in order]
            per_usa[uk] = per_line
        result[jk] = per_usa
        if i % 100 == 0:
            print(f"{i}/{len(cands)}", file=sys.stderr)

    with open("line_alignments.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"Wrote line_alignments.json ({len(result)} JPN convs)", file=sys.stderr)


if __name__ == "__main__":
    main()
