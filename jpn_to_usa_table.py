import json
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MIN_LINES = 2
TOP_K = 3
MIN_SCORE = 0.35

CLEAN_PATTERNS = [
    (r"‹TK›", ""),
    (r"｜", " "),
    (r"＃｛([^｝]*)｝‹?T?K?›?＃?", lambda m: m.group(1).split("、")[0]),
    (r"＃|｛|｝", ""),
]


def clean(text):
    for pat, repl in CLEAN_PATTERNS:
        text = re.sub(pat, repl, text)
    return text.strip()


def load_convs(path):
    raw = json.load(open(path, encoding="utf-8"))
    convs = {}
    for key, val in raw.items():
        if not val or not val[0]:
            continue
        lines = [clean(v) for k, v in sorted(val[0].items()) if clean(v)]
        if len(lines) >= MIN_LINES:
            convs[key] = lines
    return convs


def bertscore_f1(sim):
    precision = sim.max(axis=0).mean()
    recall = sim.max(axis=1).mean()
    return 2 * precision * recall / (precision + recall + 1e-9)


def main():
    usa = load_convs("voxText-usa.json")
    jpn = load_convs("voxText-jpn.json")
    print(f"USA {len(usa)} JPN {len(jpn)}", file=sys.stderr)

    model = SentenceTransformer("sentence-transformers/LaBSE")

    usa_keys = list(usa.keys())
    jpn_keys = list(jpn.keys())

    usa_line_embs = {k: model.encode(usa[k], normalize_embeddings=True) for k in usa_keys}
    jpn_line_embs = {k: model.encode(jpn[k], normalize_embeddings=True) for k in jpn_keys}

    usa_whole = model.encode([" ".join(usa[k]) for k in usa_keys], normalize_embeddings=True)
    jpn_whole = model.encode([" ".join(jpn[k]) for k in jpn_keys], normalize_embeddings=True)

    whole_sim = jpn_whole @ usa_whole.T  # (jpn, usa)

    line_sim = np.zeros_like(whole_sim)
    for j, jk in enumerate(jpn_keys):
        je = jpn_line_embs[jk]
        for i, uk in enumerate(usa_keys):
            ue = usa_line_embs[uk]
            line_sim[j, i] = bertscore_f1(je @ ue.T)

    combined = np.maximum(line_sim, whole_sim)

    rows = []
    for j, jk in enumerate(jpn_keys):
        order = np.argsort(-combined[j])[:TOP_K]
        candidates = []
        for i in order:
            score = float(combined[j, i])
            if score < MIN_SCORE:
                continue
            candidates.append(
                {
                    "usa_key": usa_keys[i],
                    "combined_score": round(score, 3),
                    "line_f1": round(float(line_sim[j, i]), 3),
                    "whole_cosine": round(float(whole_sim[j, i]), 3),
                }
            )
        rows.append({"jpn_key": jk, "jpn_lines": len(jpn[jk]), "candidates": candidates})

    with open("jpn_to_usa_candidates.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # markdown table: jpn_key | usa candidates (key:score, ...)
    with open("jpn_to_usa_table.md", "w", encoding="utf-8") as f:
        f.write("| JPN vox# | Most likely USA vox# (score) |\n")
        f.write("|---|---|\n")
        for r in rows:
            cand_str = ", ".join(f"{c['usa_key']} ({c['combined_score']})" for c in r["candidates"]) or "*(no candidate >= threshold)*"
            f.write(f"| {r['jpn_key']} | {cand_str} |\n")

    no_match = sum(1 for r in rows if not r["candidates"])
    print(f"JPN convs with no candidate >= {MIN_SCORE}: {no_match}/{len(rows)}", file=sys.stderr)
    print("Wrote jpn_to_usa_candidates.json and jpn_to_usa_table.md", file=sys.stderr)


if __name__ == "__main__":
    main()
