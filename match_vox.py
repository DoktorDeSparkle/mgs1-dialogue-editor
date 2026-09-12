import json
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment

MIN_LINES = 2

CLEAN_PATTERNS = [
    (r"‹TK›", ""),
    (r"｜", " "),
    (r"＃｛([^｝]*)｝‹?T?K?›?＃?", lambda m: m.group(1).split("、")[0]),  # ruby/alt-text blocks: keep first reading
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
        text_dict = val[0]
        lines = [clean(v) for k, v in sorted(text_dict.items()) if clean(v)]
        if len(lines) >= MIN_LINES:
            convs[key] = lines
    return convs


def bertscore_f1(sim):
    precision = sim.max(axis=0).mean()  # best USA match per JPN line
    recall = sim.max(axis=1).mean()  # best JPN match per USA line
    return 2 * precision * recall / (precision + recall + 1e-9)


def main():
    print("Loading conversations...", file=sys.stderr)
    usa = load_convs("voxText-usa.json")
    jpn = load_convs("voxText-jpn.json")
    print(f"USA convs >= {MIN_LINES} lines: {len(usa)}", file=sys.stderr)
    print(f"JPN convs >= {MIN_LINES} lines: {len(jpn)}", file=sys.stderr)

    print("Loading LaBSE...", file=sys.stderr)
    model = SentenceTransformer("sentence-transformers/LaBSE")

    usa_keys = list(usa.keys())
    jpn_keys = list(jpn.keys())

    print("Embedding USA lines...", file=sys.stderr)
    usa_embs = {k: model.encode(usa[k], normalize_embeddings=True) for k in usa_keys}
    print("Embedding JPN lines...", file=sys.stderr)
    jpn_embs = {k: model.encode(jpn[k], normalize_embeddings=True) for k in jpn_keys}

    print("Computing similarity matrix...", file=sys.stderr)
    M = np.zeros((len(usa_keys), len(jpn_keys)), dtype=np.float32)
    for i, uk in enumerate(usa_keys):
        ue = usa_embs[uk]
        for j, jk in enumerate(jpn_keys):
            je = jpn_embs[jk]
            sim = ue @ je.T
            M[i, j] = bertscore_f1(sim)

    print("Solving assignment...", file=sys.stderr)
    row_ind, col_ind = linear_sum_assignment(-M)

    results = []
    for i, j in zip(row_ind, col_ind):
        results.append(
            {
                "usa_key": usa_keys[i],
                "jpn_key": jpn_keys[j],
                "score": float(M[i, j]),
                "usa_lines": len(usa[usa_keys[i]]),
                "jpn_lines": len(jpn[jpn_keys[j]]),
                "usa_preview": usa[usa_keys[i]][0][:60],
                "jpn_preview": jpn[jpn_keys[j]][0][:60],
            }
        )

    results.sort(key=lambda r: -r["score"])
    with open(f"vox_matches_min{MIN_LINES}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    high = [r for r in results if r["score"] >= 0.5]
    low = [r for r in results if r["score"] < 0.5]
    print(f"\nTotal matched pairs: {len(results)}", file=sys.stderr)
    print(f"High confidence (>=0.5): {len(high)}", file=sys.stderr)
    print(f"Low confidence (<0.5): {len(low)}", file=sys.stderr)
    print(f"Wrote vox_matches_min{MIN_LINES}.json", file=sys.stderr)


if __name__ == "__main__":
    main()
