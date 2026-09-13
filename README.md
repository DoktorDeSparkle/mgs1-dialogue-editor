# MGS1 Undub Subtitle Workbench

Tools for the Metal Gear Solid (PS1) undub: matching Japanese and US subtitle entries,
machine-translating the Japanese script with local models, and reviewing/editing the
English subtitles before exporting them back to the game's JSON format.

It covers **vox** (codec/voice calls), **demo** (cutscenes) and **zMovie** (movie
subtitles) for both discs. All of them use the same format:

```json
"vox-0007": [
  { "01": "<line 1 text>", "02": "<line 2 text>" },   // subtitle text per line
  { "01": "0,50",          "02": "57,88" }          // "start,duration" per line
]
```

**No game data is included in this repository.** Supply your own extracted text files
(see [Data files](#data-files)).

## Setup

Requires Python 3.11+ on macOS/Linux.

```sh
python3 -m venv .venv
.venv/bin/pip install sentence-transformers scipy numpy
```

The editor server and batch translator use only the standard library. The venv is needed
for the LaBSE matching and agreement-scoring scripts.

### Data files

Copy your own extracted script JSONs into place (they are gitignored):

| Dataset | JPN | USA |
|---|---|---|
| `vox` | `voxText-jpn.json` | `voxText-usa.json` |
| `demo-d1` / `demo-d2` | `data/demo-d{1,2}/demoText-jpn.json` | `data/demo-d{1,2}/demoText-usa.json` |
| `zmovie-d1` / `zmovie-d2` | `data/zmovie-d{1,2}/zMovie-jpn.json` | `data/zmovie-d{1,2}/zMovie-usa.json` |

Paths are defined in `vox_editor/datasets.json`. All matching outputs, editor state and
exports are generated locally and also gitignored. Run the matching step (below) before
opening the editor.

### Translation engines

Engines are configured in `vox_editor/config.json`. If that file doesn't exist,
`vox_editor/config.example.json` is used, which expects **LM Studio** on
`http://127.0.0.1:1234` with two models loaded:

| Role | Model | Notes |
|---|---|---|
| Primary (draft subtitles) | `sugoi-14b-ultra` (Sugoi-14B-Ultra GGUF, Q4_K_M) | Japanese→English game-dialogue finetune |
| Alternate | `google/gemma-4-26b-a4b` | Sent with `reasoning_effort: "none"` to disable thinking |

```sh
lms load sugoi-14b-ultra -c 8192 --parallel 4
lms load google/gemma-4-26b-a4b -c 32768 --parallel 4
```

Keep the context modest. Gemma's default 262k context used enough memory to get a batch
run killed.

Other providers (`openai`, `anthropic`, `deepl`, `libretranslate`, `mock`) are supported:
copy `config.example.json` to `config.json` and edit it. Put API keys either in
`config.json`, which is gitignored, or in environment variables. Never put them in
`config.example.json`.

| Env var | Overrides (primary engine) |
|---|---|
| `VOX_TRANSLATE_PROVIDER` | provider |
| `VOX_TRANSLATE_URL` | endpoint URL |
| `VOX_TRANSLATE_MODEL` | model name |
| `VOX_TRANSLATE_API_KEY` | API key |

The engine choice came from a 49-line bake-off of Sugoi v4, Sugoi-14B-Ultra, Qwen 3.8 27B
and Gemma 4 against the US dub (`mt_bakeoff/bakeoff.py`; summary in `notes/CLAUDE.md`).

## Workflow

### 1. Match JPN ↔ USA entries

Vox conversation IDs don't line up between releases, so they're matched with LaBSE
cross-lingual embeddings:

```sh
.venv/bin/python match_vox.py            # line-alignment F1 + Hungarian 1:1 → vox_matches*.json
.venv/bin/python match_vox2.py           # line-F1 vs whole-conversation cosine → vox_dual_scores.json
.venv/bin/python jpn_to_usa_table.py     # top-3 USA candidates per JPN conv → jpn_to_usa_candidates.json
.venv/bin/python line_alignments.py      # per-line alignments for highlighting → line_alignments.json
```

Demo and zMovie IDs already correspond, but the same candidates and alignments are built for them:

```sh
.venv/bin/python match_dataset.py demo-d1 demo-d2 zmovie-d1 zmovie-d2
```

### 2. Machine-translate everything

```sh
./run_all_mt.sh                                          # every dataset, both engines, then agreement scoring
.venv/bin/python vox_editor/batch_translate.py --dataset demo-d1          # one dataset
.venv/bin/python vox_editor/batch_translate.py --dataset vox --only vox-0007
.venv/bin/python vox_editor/batch_translate.py --dataset vox --score      # LaBSE agreement between engines
```

- The pass is resumable: chunks that already have a translation are skipped.
- It's safe to run while the editor is open, because state writes share a file lock.
- Lines you've edited or picked in review are never overwritten.
- The agreement score is the LaBSE similarity between the two engines' English. Below
  **0.70** they usually mean different things, so review those lines first.

### 3. Review and edit

```sh
.venv/bin/python vox_editor/server.py     # http://127.0.0.1:8765
```

- **Editor** (`/?ds=vox`, dataset switcher in the top bar):
  - The JPN entry is on the left. Shift-click lines and press **G** to group a split
    sentence into one chunk.
  - Edit, split (⌘Enter), insert or re-time subtitles.
  - The matched USA script is on the right, with aligned lines highlighted.
  - Sort by **MT engines disagree most** to find problem lines.
- **A/B review** (`/review.html?ds=vox`):
  - Shows one line at a time, with the lead-up context, the next line and the aligned US dub line.
  - **A** = Sugoi, **B** = Gemma, **E** = write your own, **S** = skip, **⌫** = undo.
  - **Finish for now** returns to the editor. Every choice is saved immediately.

### 4. Export

**Export JSON** in the editor writes the dataset's format to `exports/`, e.g.
`exports/demoText-jpn-undub-d1.json`:

- Edited entries get their English subtitles, renumbered `01..NN` with timings.
- Untouched entries, and chunks without subtitles, keep the original Japanese.
- The line break inside a subtitle is set per dataset: vox uses `\r`, demo and zMovie use `｜`.

## Layout

| Path | What |
|---|---|
| `voxText-{jpn,usa}.json` | Vox source text, disc 1 (not included; you supply it) |
| `data/<dataset>/` | Demo/zMovie source text (not included; you supply it) |
| `vox_editor/datasets.json` | Dataset definitions: source files, state, export name, line break |
| `vox_editor_state.json`, `state/` | Editor state: chunks, translations from both engines, decisions, subtitles (generated) |
| `match/`, `jpn_to_usa_candidates.json`, `line_alignments.json`, `vox_*.json` | Matching outputs (generated) |
| `vox_editor/server.py` | Local web server (stdlib), translation proxy, export |
| `vox_editor/batch_translate.py` | Batch translation and agreement scoring |
| `vox_editor/subtitles.py` | Markup cleanup, subtitle wrapping/splitting, timing (mirrors `static/app.js`) |
| `vox_editor/static/` | Editor (`index.html`, `app.js`) and A/B review (`review.html`) |
| `exports/` | Exported undub JSON |
| `mt_bakeoff/` | Translation model comparison script (results stay local) |
| `notes/` | Development notes (`notes/CLAUDE.md`: method details, decisions, open threads) |

## Game markup

| Markup | Meaning | Handling |
|---|---|---|
| `‹TK›`, `‹BK›` | Text control codes | Stripped for matching/MT |
| `｜` (and `\r` in USA vox) | Line break within a subtitle | Newline in the editor, restored on export |
| `＃｛base、reading｝＃` (vox) / `#｛base、reading｝#` (demo) | Ruby / alt text | Shown as ruby. MT gets `base（reading）`, or just `base` when it's already Latin (e.g. FOX HOUND) |

## Notes

- The repository contains tooling only. Game script text, translations of it, and anything
  generated from it stay local (see `.gitignore`).
- Subtitle row width is 40 characters, from the US data (p95 = 39).
- Timing units are the game's own: `"start,duration"`.
