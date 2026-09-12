# undub-vox-vector-comparison

Matching USA and JPN vox (voice/subtitle) conversations for the MGS1 undub project,
using multilingual sentence embeddings — since conversation IDs (`vox-NNNN`) don't
correspond 1:1 between the two releases, and subtitle line breaks/content diverge
due to localization.

## Data

- `voxText-usa.json`, `voxText-jpn.json` — keyed by `vox-XXXX`. Each value is
  `[textDict, timingDict]`. `textDict` maps `"01".."NN"` → subtitle line text.
  Game markup to strip: `‹TK›`, `｜` (line break), `＃｛ruby｝＃` (furigana/alt-text blocks).
- USA: 1105 keys, 1043 non-empty. JPN: 1123 keys, 1055 non-empty.
- Most entries are 1-2 line barks (warnings, one-off calls) — not useful for
  vector conversation-matching; only worth doing this for entries with >=2-3 lines.
  - USA >=3 lines: 309, JPN >=3 lines: 288
  - USA >=2 lines: 523, JPN >=2 lines: 460

## Method

Model: `sentence-transformers/LaBSE` (cross-lingual JP/EN sentence embeddings).

Two complementary scoring methods, computed per USA×JPN conversation pair:

1. **Line-alignment F1** (`match_vox.py`) — BERTScore-style: embed each line
   individually, for each JPN line take max cosine sim to any USA line (avg =
   precision), and vice versa (avg = recall), F1 = harmonic mean. Robust to
   re-segmentation/reordering across localization. Then run Hungarian
   assignment (`scipy.optimize.linear_sum_assignment`) for a global 1:1 match.
   Output: `vox_matches.json` (min 3 lines), `vox_matches_min2.json` (min 2 lines).

2. **Whole-conversation cosine** (`match_vox2.py`) — concatenate all lines per
   conversation into one string, embed once, cosine similarity. Smooths over
   single divergent lines (e.g. localized idiom swaps) but is more forgiving of
   topically-similar-but-wrong matches. Output: `vox_dual_scores.json` — for
   each USA conv, independently finds best JPN match under *each* method
   (not a shared 1:1 assignment) and flags `agree: true/false`.

### Confidence signal

Where line-F1's top pick and whole-conversation cosine's top pick **agree**,
treat as high confidence. At min 2 lines: **427/523 (81.6%) agree**. The 96
disagreements are the priority manual-review list — one method or the other
is likely being fooled (e.g. localized idiom breaking line-alignment, or
topical overlap fooling whole-conversation cosine).

### Known failure mode

Localization sometimes swaps in an unrelated idiom/reference (e.g. a JPN
proverb replaced with a Shakespeare quote in USA) instead of translating
literally. This tanks line-alignment score for that one line even when the
rest of the conversation matches, since LaBSE has no semantic bridge between
unrelated idioms. Whole-conversation cosine is more forgiving of this
specific case (see vox-0612 USA / vox-0622 JPN: line-F1 0.401, whole-conv
0.548) — another reason to use both scores together rather than relying on
line-F1 alone.

## Files

- `match_vox.py` — line-alignment F1 + Hungarian assignment (main matcher, min-lines configurable via `MIN_LINES`)
- `match_vox2.py` — dual scoring (line-F1 vs whole-conversation cosine), non-exclusive top-1 per USA conv, agreement flag
- `vox_matches.json` / `vox_matches_min2.json` — line-F1 Hungarian-assigned matches, sorted by score desc
- `vox_dual_scores.json` — per-USA-conv dual scores + agreement flag
- `jpn_to_usa_table.py` → `jpn_to_usa_candidates.json` / `jpn_to_usa_table.md` — per-JPN-conv top-3 USA candidates (max of line-F1, whole-cosine)
- `line_alignments.py` → `line_alignments.json` — per-JPN-line top-2 USA line matches within each candidate (drives GUI highlighting)
- `vox_editor/` — undub subtitle editor (first draft). `.venv/bin/python vox_editor/server.py` → http://127.0.0.1:8765.
  Stdlib server + vanilla JS. JPN conv on the left (group lines into chunks, MT via proxied endpoint, edit/split/insert
  subtitles with timings); USA reference on the right, pre-selected from vector candidates. Edit state lives in
  `vox_editor_state.json`; export writes `voxText-jpn-en.json` (same `[textDict, timingDict]` format, timings `"start,dur"`).
  MT engines in `vox_editor/config.json` (copy `config.example.json`): `primary` engine's output drives draft subtitles,
  other engines are stored per chunk as `mts` alternates. Current setup: Sugoi-14B-Ultra (primary) + Gemma 4 26B-A4B,
  both via LM Studio (`reasoning_effort: "none"` disables thinking). Chosen via a 49-line bake-off vs Qwen 3.8 27B and
  Sugoi v4: Qwen reversed negations, Sugoi v4 too error-prone.
- `vox_editor/batch_translate.py` — full resumable MT pass into the editor state (conversations get status `mt`
  = MT draft; never overwrites convs already touched in the editor). `--score` then computes LaBSE similarity between
  engines' English per chunk (`mt_agree`); the editor's "MT engines disagree most" sort uses it as the review queue.
- `vox_editor/static/review.html` — A/B review stepper (http://127.0.0.1:8765/review.html, "A/B review →" in the editor):
  lead-up context + next line + aligned US dub line, keys A/B/E(dit)/S(kip)/⌫(undo), "Finish for now" returns to the
  editor. Decisions go through `/api/decide`, which patches the on-disk state under the lock (never a stale whole conv).
  Chunk flags: `decided` / `mtEdited` / `subsEdited` / `timingEdited` lock a chunk against batch overwrites.
- **Datasets** (`vox_editor/datasets.json`): `vox`, `demo-d1`, `demo-d2`, `zmovie-d1`, `zmovie-d2` — all the same
  `[textDict, timingDict]` format. Demo/zMovie snapshots live in `data/` (copied from `mgs1-undub/workingFiles/`),
  per-dataset editor state in `state/` (vox keeps `vox_editor_state.json`), exports go to `exports/`. Demo/zMovie keys
  correspond 1:1 JPN↔USA (`same_id`), so the USA reference defaults to the same key; `match_dataset.py` still builds
  candidates + line alignments into `match/`. Demo JSON writes ruby as half-width `#｛base、reading｝#`. Export line break
  is per dataset (vox `\r`, demo/zMovie `｜`). Editor/review take `?ds=`; batch takes `--dataset`; `run_all_mt.sh` runs
  every dataset then scores agreement.
- `mt_bakeoff/` — record of the translation model comparison (script, raw results, HTML report).
- `.venv/` — local venv with `sentence-transformers`, `scipy`, `numpy`

## Open threads / next steps

- 1-2 line "barks" (~600+ per side) still unmatched — line-alignment/whole-conv
  cosine likely won't disambiguate near-duplicate short lines ("Snake!") well;
  probably needs a different approach (positional/sequential context relative
  to already-matched neighboring conversations, rather than pure text similarity).
- Haven't yet built a combined/blended ranking score — decided to keep the two
  scores separate and use agreement/disagreement as the review signal instead
  of a naive sum, since it's unclear that summing is meaningful.
- Manual review pass on the 96 disagreement cases not yet started.
