"""Subtitle text helpers shared by server.py, batch_translate.py and review decisions.
Python ports of the same functions in static/app.js — keep them in sync."""
import re

# Subtitle limits from mgs-qt-ui (mgs_font_text.py / FEATURES.md): 260px per row; radio calls 4 rows, in-game 2.
# MGS1 ASCII glyph widths (original_widths.txt) from " " (0x20) to "~" (0x7E); kana/kanji glyphs are 12px.
MAX_PX, RADIO_ROWS, SCENE_ROWS, WIDE_PX = 260, 4, 2, 12
GLYPH_PX = [4, 5, 5, 12, 7, 10, 8, 3, 3, 3, 5, 8, 3, 5, 3, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 4, 8, 4, 8,
            5, 8, 9, 9, 9, 8, 8, 10, 9, 4, 7, 9, 8, 11, 9, 10, 8, 10, 9, 8, 8, 9, 8, 12, 8, 8, 8, 3, 8, 3, 4, 6,
            3, 7, 8, 7, 8, 8, 4, 8, 7, 3, 3, 7, 3, 10, 7, 7, 8, 8, 4, 6, 4, 7, 6, 9, 6, 6, 6, 2, 2, 2, 5]

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


def px_width(text):
    return sum(GLYPH_PX[ord(c) - 0x20] if 0x20 <= ord(c) <= 0x7E else WIDE_PX if ord(c) > 0x7E else 0 for c in text)


def wrap_lines(text):
    """Greedy word wrap at MAX_PX (as mgs_font_text.wrap_text), then rebalance a 2-row result to the most even break."""
    rows, row = [], ""
    for w in " ".join(text.split()).split(" "):
        if row and px_width(f"{row} {w}") > MAX_PX:
            rows.append(row)
            row = w
        else:
            row = f"{row} {w}" if row else w
    if row:
        rows.append(row)
    if len(rows) == 2:
        t, best = " ".join(rows), None
        for i, ch in enumerate(t):
            if ch == " ":
                a, b = px_width(t[:i]), px_width(t[i + 1:])
                if a <= MAX_PX and b <= MAX_PX and (best is None or abs(a - b) < best[0]):
                    best = (abs(a - b), i)
        if best:
            return [t[:best[1]], t[best[1] + 1:]]
    return rows


def wrap_rows(text):
    return "\n".join(wrap_lines(text))


def split_pieces(text, rows=SCENE_ROWS):
    """Split a translation into subtitle-sized pieces (<= rows rows each), preferring sentence boundaries."""
    text = " ".join(text.split())
    fits = lambda s: len(wrap_lines(s)) <= rows  # noqa: E731
    if fits(text):
        return [text]
    pieces, acc = [], ""
    for s in re.split(r"(?<=[.!?…—])\s+", text):
        while not fits(s):  # overlong sentence: take what fits, cutting after a comma in the second half
            head = " ".join(wrap_lines(s)[:rows])
            comma = head.rfind(", ")
            cut = comma + 1 if comma >= len(head) * 0.4 else len(head)
            if acc:
                pieces.append(acc)
                acc = ""
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if acc and not fits(f"{acc} {s}"):
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


def auto_subs(chunk, text, timings, rows=SCENE_ROWS):
    subs = [{"text": wrap_rows(p), "start": 0, "dur": 0} for p in split_pieces(text, rows)]
    start, end = chunk_span(chunk, timings)
    total = sum(max(len(s["text"]), 1) for s in subs) or 1
    t = start
    for i, s in enumerate(subs):
        d = end - t if i == len(subs) - 1 else round((end - start) * max(len(s["text"]), 1) / total)
        s["start"], s["dur"] = round(t), max(1, round(d))
        t += d
    chunk.update(subs=subs, timingEdited=False, subsEdited=False)


