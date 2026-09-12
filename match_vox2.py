import json
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MIN_LINES = 2

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

    whole_sim = usa_whole @ jpn_whole.T  # (len(usa_keys), len(jpn_keys))

    line_sim = np.zeros_like(whole_sim)
    for i, uk in enumerate(usa_keys):
        ue = usa_line_embs[uk]
        for j, jk in enumerate(jpn_keys):
            je = jpn_line_embs[jk]
            line_sim[i, j] = bertscore_f1(ue @ je.T)

    results = []
    for i, uk in enumerate(usa_keys):
        j_line = int(np.argmax(line_sim[i]))
        j_whole = int(np.argmax(whole_sim[i]))
        results.append(
            {
                "usa_key": uk,
                "line_best_jpn": jpn_keys[j_line],
                "line_best_score": float(line_sim[i, j_line]),
                "whole_best_jpn": jpn_keys[j_whole],
                "whole_best_score": float(whole_sim[i, j_whole]),
                "agree": jpn_keys[j_line] == jpn_keys[j_whole],
            }
        )

    with open("vox_dual_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    agree = sum(r["agree"] for r in results)
    print(f"Agree: {agree}/{len(results)} ({100*agree/len(results):.1f}%)", file=sys.stderr)
    print("Wrote vox_dual_scores.json", file=sys.stderr)


if __name__ == "__main__":
    main()
