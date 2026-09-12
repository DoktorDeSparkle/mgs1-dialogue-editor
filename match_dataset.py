"""LaBSE matching for any dataset in vox_editor/datasets.json (demo, zMovie, ...).

    .venv/bin/python match_dataset.py demo-d1 demo-d2 zmovie-d1 zmovie-d2

Writes the dataset's `candidates` (top-3 USA entries per JPN entry, max of
line-F1 and whole-entry cosine — same method as jpn_to_usa_table.py) and
`alignments` (per-JPN-line top-2 USA lines within each candidate — same as
line_alignments.py). Unlike vox, every non-empty entry is included (MIN_LINES=1),
since demo/zMovie IDs already correspond and short entries still need alignment.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
TOP_K, TOP_N, MIN_SCORE = 3, 2, 0.35


def clean(text):
    text = re.sub(r"‹[A-Z]+›", "", text)
    text = re.sub(r"[｜\r\n]", " ", text)
    text = re.sub(r"[#＃]｛([^｝]*)｝[#＃]?", lambda m: m.group(1).split("、")[0], text)
    return re.sub(r"[#＃｛｝]", "", text).strip()


def load_lines(path):
    out = {}
    for key, val in json.loads(Path(path).read_text(encoding="utf-8")).items():
        if not val or not val[0]:
            continue
        pairs = [(k, clean(v)) for k, v in sorted(val[0].items())]
        pairs = [(k, t) for k, t in pairs if t]
        if pairs:
            out[key] = pairs
    return out


def bertscore_f1(sim):
    p, r = sim.max(axis=0).mean(), sim.max(axis=1).mean()
    return 2 * p * r / (p + r + 1e-9)


def run(name, cfg, model):
    jpn, usa = load_lines(ROOT / cfg["jpn"]), load_lines(ROOT / cfg["usa"])
    jk, uk = list(jpn), list(usa)
    enc = lambda lines: model.encode([t for _, t in lines], normalize_embeddings=True)  # noqa: E731
    je, ue = {k: enc(jpn[k]) for k in jk}, {k: enc(usa[k]) for k in uk}
    whole_j = model.encode([" ".join(t for _, t in jpn[k]) for k in jk], normalize_embeddings=True)
    whole_u = model.encode([" ".join(t for _, t in usa[k]) for k in uk], normalize_embeddings=True)
    whole = whole_j @ whole_u.T
    line = np.array([[bertscore_f1(je[a] @ ue[b].T) for b in uk] for a in jk])
    combined = np.maximum(line, whole)

    rows, aligns = [], {}
    for i, a in enumerate(jk):
        cands = []
        for j in np.argsort(-combined[i])[:TOP_K]:
            if combined[i, j] >= MIN_SCORE or uk[j] == a:
                cands.append({"usa_key": uk[j], "combined_score": round(float(combined[i, j]), 3),
                              "line_f1": round(float(line[i, j]), 3), "whole_cosine": round(float(whole[i, j]), 3)})
        if cfg.get("same_id") and a in usa and all(c["usa_key"] != a for c in cands):
            j = uk.index(a)
            cands.append({"usa_key": a, "combined_score": round(float(combined[i, j]), 3),
                          "line_f1": round(float(line[i, j]), 3), "whole_cosine": round(float(whole[i, j]), 3)})
        rows.append({"jpn_key": a, "jpn_lines": len(jpn[a]), "candidates": cands})
        aligns[a] = {}
        for c in cands:
            sim = je[a] @ ue[c["usa_key"]].T
            aligns[a][c["usa_key"]] = {jpn[a][li][0]: [[usa[c["usa_key"]][o][0], round(float(sim[li, o]), 3)]
                                                      for o in sim[li].argsort()[::-1][:TOP_N]]
                                       for li in range(len(jpn[a]))}
    Path(ROOT / cfg["candidates"]).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["candidates"]).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / cfg["alignments"]).write_text(json.dumps(aligns, ensure_ascii=False), encoding="utf-8")
    same = sum(1 for r in rows if r["candidates"] and r["candidates"][0]["usa_key"] == r["jpn_key"])
    print(f"{name}: {len(rows)} JPN entries, top candidate == same ID for {same}; "
          f"not: {[(r['jpn_key'], r['candidates'][0]['usa_key'] if r['candidates'] else None) for r in rows if not r['candidates'] or r['candidates'][0]['usa_key'] != r['jpn_key']]}")


if __name__ == "__main__":
    datasets = json.loads((ROOT / "vox_editor/datasets.json").read_text())
    model = SentenceTransformer("sentence-transformers/LaBSE")
    for name in sys.argv[1:]:
        run(name, datasets[name], model)
