"""MT bake-off: Sugoi v4 (CT2), Sugoi-14B-Ultra, Qwen3.8-27B, Gemma-4-26B vs the USA dub script.
Run with scratchpad mtvenv python from the project dir."""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SP = Path(__file__).parent
PROJ = Path("/Users/solidmixer/projects/undub-vox-vector-comparison")
sys.path.insert(0, str(PROJ / "vox_editor"))
from server import SYSTEM_PROMPT, build_user_prompt  # the editor's actual prompt

CONVS = ["vox-0622", "vox-0850", "vox-0472", "vox-1037", "vox-0007"]
LM = "http://127.0.0.1:1234/v1/chat/completions"
SUGOI_SYS = ("You are a professional localizer whose primary goal is to translate Japanese to English. "
             "You should use colloquial or slang or nsfw vocabulary if it makes the translation more accurate. Always respond in English.")

jpn = json.loads((PROJ / "voxText-jpn.json").read_text())
usa = json.loads((PROJ / "voxText-usa.json").read_text())
cands = {r["jpn_key"]: r for r in json.loads((PROJ / "jpn_to_usa_candidates.json").read_text())}
align = json.loads((PROJ / "line_alignments.json").read_text())


def plain(raw):
    s = re.sub(r"‹[A-Z]+›", "", raw)
    s = re.sub(r"＃｛([^｝]*)｝＃?", lambda m: "（".join(m.group(1).split("、", 1)) + ("）" if "、" in m.group(1) else ""), s)
    return re.sub(r"[＃｛｝]", "", s).replace("｜", " ").strip()


def usa_text(raw):
    return re.sub(r"\s*(\r\n?|｜)\s*", " ", raw).strip()


def chat(model, system, user, temperature, extra):
    body = {"model": model, "temperature": temperature, "max_tokens": 400,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **extra}
    r = urllib.request.Request(LM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(r, timeout=600))
    txt = re.sub(r"<think>.*?</think>", "", out["choices"][0]["message"]["content"], flags=re.S).strip()
    return txt, out.get("usage", {})


# ---- build jobs
rows = []
for ck in CONVS:
    keys = sorted(jpn[ck][0])
    lines = [plain(jpn[ck][0][k]) for k in keys]
    uk = cands[ck]["candidates"][0]["usa_key"]
    for i, k in enumerate(keys):
        al = align.get(ck, {}).get(uk, {}).get(k, [])
        rows.append({"conv": ck, "usa_key": uk, "line": k, "ja": lines[i],
                     "usa_aligned": [[ul, sim, usa_text(usa[uk][0][ul])] for ul, sim in al[:1]],
                     "_ctx_full": "\n".join(("▶ " if j == i else "  ") + t for j, t in enumerate(lines)),
                     "_ctx_10": "\n".join(("▶ " if j == i else "  ") + lines[j] for j in range(max(0, i - 5), min(len(lines), i + 5)))})

models = {
    "qwen3.8-27b": dict(model="qwen/qwen3.8-27b", system=SYSTEM_PROMPT, ctx="_ctx_full", temperature=0.3, extra={"reasoning_effort": "none"}),
    "gemma-4-26b-a4b": dict(model="google/gemma-4-26b-a4b", system=SYSTEM_PROMPT, ctx="_ctx_full", temperature=0.3, extra={"reasoning_effort": "none"}),
    "sugoi-14b-ultra": dict(model=sys.argv[1] if len(sys.argv) > 1 else "sugoi-14b-ultra", system=SUGOI_SYS, ctx="_ctx_10", temperature=0.1,
                            extra={"top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1}),
}
timing = {}

# ---- Sugoi v4 (CTranslate2, sentence-level, no context)
import ctranslate2, sentencepiece as spm
t = time.time()
tr = ctranslate2.Translator(str(SP / "sugoi-v4"), device="cpu", compute_type="int8", inter_threads=4)
sp_ja = spm.SentencePieceProcessor(model_file=str(SP / "sugoi-v4/spm/spm.ja.nopretok.model"))
sp_en = spm.SentencePieceProcessor(model_file=str(SP / "sugoi-v4/spm/spm.en.nopretok.model"))
res = tr.translate_batch([sp_ja.encode(r["ja"], out_type=str) for r in rows], beam_size=5, max_decoding_length=256)
for r, out in zip(rows, res):
    r["sugoi-v4"] = sp_en.decode(out.hypotheses[0]).replace("<unk>", "").strip()
timing["sugoi-v4"] = round(time.time() - t, 1)
print("sugoi-v4 done", timing["sugoi-v4"], file=sys.stderr)


def run_model(name):
    cfg = models[name]
    t = time.time()

    def one(r):
        user = build_user_prompt({"text": r["ja"], "context": r[cfg["ctx"]]})
        try:
            txt, usage = chat(cfg["model"], cfg["system"], user, cfg["temperature"], cfg["extra"])
        except Exception as e:
            txt, usage = f"ERROR {e}", {}
        r[name] = txt
        return usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

    with ThreadPoolExecutor(4) as ex:  # 4 parallel requests per model
        reasoning = sum(ex.map(one, rows))
    timing[name] = round(time.time() - t, 1)
    print(name, "done", timing[name], "s, reasoning tokens:", reasoning, file=sys.stderr)


with ThreadPoolExecutor(3) as ex:
    list(ex.map(run_model, [m for m in models if m in sys.argv[2:] or len(sys.argv) <= 2]))

prev = json.loads((SP / "bakeoff.json").read_text()) if (SP / "bakeoff.json").exists() else None
for i, r in enumerate(rows):
    for k in [k for k in r if k.startswith("_")]:
        del r[k]
    if prev:  # merge results from earlier partial runs
        for k, v in prev["rows"][i].items():
            r.setdefault(k, v)
if prev:
    timing = {**prev["timing"], **timing}
out = {"rows": rows, "timing": timing, "n_lines": len(rows),
       "usa_full": {c: [[k, usa_text(v)] for k, v in sorted(usa[cands[c]["candidates"][0]["usa_key"]][0].items())] for c in CONVS},
       "scores": {c: cands[c]["candidates"][0] for c in CONVS}}
(SP / "bakeoff.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote bakeoff.json", timing, file=sys.stderr)
