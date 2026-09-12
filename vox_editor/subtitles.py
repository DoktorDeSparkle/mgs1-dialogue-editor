"""Subtitle text helpers shared by server.py, batch_translate.py and review decisions.
Python ports of the same functions in static/app.js — keep them in sync."""
import re

ROW_WIDTH, MAX_ROWS = 40, 2  # keep in sync with static/app.js

def jpn_plain(raw):
    s = re.sub(r"‹[A-Z]+›", "", raw)

    def ruby(m):
        base, _, reading = m.group(1).partition("、")
        # a Latin base (FOX HOUND) is already the term; a kanji base keeps its reading
        return f"{base}（{reading}）" if reading and not re.search(r"[A-Za-z]", base) else base

    s = re.sub(r"[#＃]｛([^｝]*)｝[#＃]?", ruby, s)  # vox uses ＃, demo uses #
    return re.sub(r"[｜\r\n]", " ", re.sub(r"[#＃｛｝]", "", s)).strip()


def needs_mt(text):
    """Lines that are only punctuation ("……", "！？") pass through untranslated."""
    return bool(re.search(r"[぀-ヿ㐀-鿿A-Za-zＡ-Ｚａ-ｚ0-9０-９]", text))


def wrap_rows(text):
    text = " ".join(text.split())
    if len(text) <= ROW_WIDTH:
        return text
    if len(text) <= ROW_WIDTH * 2:
        mid, best = len(text) / 2, -1
        for i, ch in enumerate(text):
            if ch == " " and i <= ROW_WIDTH and len(text) - i - 1 <= ROW_WIDTH:
                if best < 0 or abs(i - mid) < abs(best - mid):
                    best = i
        if best > 0:
            return text[:best] + "\n" + text[best + 1:]
    rows, row = [], ""
    for w in text.split(" "):
        if row and len(row) + 1 + len(w) > ROW_WIDTH:
            rows.append(row)
            row = w
        else:
            row = f"{row} {w}" if row else w
    if row:
        rows.append(row)
    return "\n".join(rows)


def split_pieces(text):
    text = " ".join(text.split())
    cap = ROW_WIDTH * MAX_ROWS
    if len(text) <= cap:
        return [text]
    pieces, acc = [], ""
    for s in re.split(r"(?<=[.!?…—])\s+", text):
        while len(s) > cap:
            cut = s.rfind(", ", 0, cap + 1)
            if cut < cap * 0.4:
                cut = s.rfind(" ", 0, cap + 1)
            if cut <= 0:
                cut = cap
            if acc:
                pieces.append(acc)
                acc = ""
            pieces.append(s[:cut + (1 if s[cut:cut + 1] == "," else 0)].strip())
            s = s[cut + 1:].strip()
        if acc and len(acc) + 1 + len(s) > cap:
            pieces.append(acc)
            acc = s
        else:
            acc = f"{acc} {s}" if acc else s
    if acc:
        pieces.append(acc)
    return pieces


def parse_timing(s):
    try:
        a, b = (int(float(x)) for x in str(s or "0,0").split(","))
        return a, b
    except ValueError:
        return 0, 0


def chunk_span(chunk, timings):
    spans = [parse_timing(timings.get(k)) for k in chunk["lines"]]
    if not spans:
        return 0, 0
    return min(s for s, _ in spans), max(s + d for s, d in spans)


def auto_subs(chunk, text, timings):
    subs = [{"text": wrap_rows(p), "start": 0, "dur": 0} for p in split_pieces(text)]
    start, end = chunk_span(chunk, timings)
    total = sum(max(len(s["text"]), 1) for s in subs) or 1
    t = start
    for i, s in enumerate(subs):
        d = end - t if i == len(subs) - 1 else round((end - start) * max(len(s["text"]), 1) / total)
        s["start"], s["dur"] = round(t), max(1, round(d))
        t += d
    chunk.update(subs=subs, timingEdited=False, subsEdited=False)


