"""Async collaboration between separate local editors via patch files.

Each collaborator runs their own server and state. Finished entries travel as
*.undub-patch.json files (send them however you like: chat, email, a shared folder):

  export  -> one patch holding one or more entries (conversation keys) of one dataset,
             with the original JPN lines they replace, the edited conv and what it exports as
  import  -> the patch lands in collab/inbox/ unreviewed; nothing touches the state yet
  review  -> collab.html shows each entry: JPN / your current version / incoming
  accept  -> the incoming conv replaces yours (the replaced conv is kept so it can be reverted)
  reject  -> nothing changes; the note is kept
  revert  -> undo an accept (if the entry wasn't edited since) or reopen a rejected entry

Conflict detection: every conv that came in through a patch remembers the hash of its
content (conv["collab"]["hash"]). A patch carries the sender's base hash, so the receiver
can tell "built on top of your current version" from "you both changed this entry".

Layout under collab/ (gitignored, it holds game text):
  inbox/    imported patches; each entry gets a "resolution" once reviewed
  archive/  patches whose entries are all resolved
  outbox/   copies of the patches you exported
"""
import datetime
import hashlib
import json
import re
import uuid
from pathlib import Path

FORMAT = "mgs1-undub-patch"
VERSION = 1
SUFFIX = ".undub-patch.json"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha1(obj):
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def content_hash(conv):
    """Hash of what a conv exports as (chunking + subtitles), ignoring MT drafts and bookkeeping."""
    if not conv:
        return None
    return sha1([[c["lines"], [[s.get("text", ""), s.get("start", 0), s.get("dur", 0)] for s in c.get("subs", [])]]
                 for c in conv.get("chunks", [])])


def has_human_work(conv):
    """False for convs that are empty or only hold untouched batch MT drafts."""
    if not conv:
        return False
    if conv.get("collab"):
        return True
    return any(c.get(f) for c in conv.get("chunks", []) for f in ("subsEdited", "timingEdited", "mtEdited", "decided"))


def slug(s, n=24):
    return re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")[:n] or "anon"


def export_rows(conv, orig, line_break):
    """The [textDict, timingDict] this conv exports as (same rules as server.export)."""
    rows = []
    for chunk in conv.get("chunks", []):
        subs = [s for s in chunk.get("subs", []) if s.get("text", "").strip()]
        if subs:
            rows += [(s["text"].strip().replace("\n", line_break), f"{int(round(s['start']))},{int(round(s['dur']))}") for s in subs]
        else:
            rows += [(orig[0][k], orig[1].get(k, "0,0")) for k in chunk["lines"] if k in orig[0]]
    return [{f"{i:02d}": t for i, (t, _) in enumerate(rows, 1)},
            {f"{i:02d}": tm for i, (_, tm) in enumerate(rows, 1)}]


def build_patch(ds, keys, state, jpn, author, note="", line_break="\r"):
    entries = []
    for key in keys:
        conv = state.get("convs", {}).get(key)
        if not conv or key not in jpn:
            raise ValueError(f"{key}: nothing to export (no saved work for this entry)")
        if not any(s.get("text", "").strip() for c in conv.get("chunks", []) for s in c.get("subs", [])):
            raise ValueError(f"{key}: has no English subtitles yet")
        entries.append({
            "key": key,
            "replaces": {"jpn": jpn[key], "jpn_hash": sha1(jpn[key])},
            "base_hash": (conv.get("collab") or {}).get("hash"),
            "hash": content_hash(conv),
            "status": conv.get("status"),
            "conv": conv,
            "export": export_rows(conv, jpn[key], line_break),
        })
    created = now()
    patch = {"format": FORMAT, "version": VERSION, "id": uuid.uuid4().hex[:12], "dataset": ds,
             "author": author, "created": created, "note": note, "entries": entries}
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    what = keys[0] if len(keys) == 1 else f"{keys[0]}--{keys[-1]}_{len(keys)}-entries"
    return patch, f"{ds}_{what}_{slug(author)}_{stamp}{SUFFIX}"


def validate(patch):
    if not isinstance(patch, dict) or patch.get("format") != FORMAT:
        raise ValueError("not an undub patch file")
    if patch.get("version", 0) > VERSION:
        raise ValueError(f"patch version {patch['version']} is newer than this editor supports ({VERSION}); update the editor")
    for f in ("id", "dataset", "entries"):
        if not patch.get(f):
            raise ValueError(f"patch is missing {f!r}")
    for e in patch["entries"]:
        if not e.get("key") or not isinstance(e.get("conv"), dict) or not isinstance(e["conv"].get("chunks"), list):
            raise ValueError("patch entry is malformed")
        for c in e["conv"]["chunks"]:
            if not isinstance(c.get("lines"), list) or not isinstance(c.get("subs", []), list):
                raise ValueError(f"{e['key']}: malformed chunk")


def assess(entry, local, jpn_entry):
    """How an incoming entry relates to the receiver's data and current version."""
    problems = []
    if jpn_entry is None:
        problems.append(f"{entry['key']} doesn't exist in your source text")
    else:
        incoming_lines = [k for c in entry["conv"]["chunks"] for k in c["lines"]]
        missing = [k for k in incoming_lines if k not in jpn_entry[0]]
        if missing:
            problems.append(f"lines {', '.join(missing)} don't exist in your copy of {entry['key']}")
    source_differs = jpn_entry is not None and entry.get("replaces", {}).get("jpn_hash") != sha1(jpn_entry)
    local_hash = content_hash(local)
    if local_hash and local_hash == entry.get("hash"):
        relation = "same"        # already identical
    elif not has_human_work(local):
        relation = "new"         # you only have MT drafts (or nothing) here
    elif entry.get("base_hash") and entry["base_hash"] == local_hash:
        relation = "update"      # built on top of exactly what you have
    else:
        relation = "conflict"    # you both changed this entry
    return {"relation": relation, "source_differs": source_differs, "problems": problems}


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def dir(self, name):
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_outbox(self, patch, filename):
        (self.dir("outbox") / filename).write_text(json.dumps(patch, ensure_ascii=False, indent=1), encoding="utf-8")

    def import_patch(self, patch, filename=""):
        validate(patch)
        for d in ("inbox", "archive"):
            for f in self.dir(d).glob(f"*{SUFFIX}"):
                if json.loads(f.read_text(encoding="utf-8")).get("id") == patch["id"]:
                    return {"duplicate": True, "file": f.name, "where": d}
        name = Path(filename).name if filename.endswith(SUFFIX) else f"{patch['dataset']}_{patch['id']}{SUFFIX}"
        dest = self.dir("inbox") / name
        if dest.exists():
            dest = dest.with_name(dest.name.replace(SUFFIX, f"_{patch['id']}{SUFFIX}"))
        patch["imported"] = now()
        dest.write_text(json.dumps(patch, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"duplicate": False, "file": dest.name, "entries": len(patch["entries"]), "dataset": patch["dataset"]}

    def patches(self, include_archive=False):
        dirs = ["inbox"] + (["archive"] if include_archive else [])
        out = []
        for d in dirs:
            for f in sorted(self.dir(d).glob(f"*{SUFFIX}")):
                p = json.loads(f.read_text(encoding="utf-8"))
                p["_file"], p["_where"] = f.name, d
                out.append(p)
        return out

    def find(self, file):
        for d in ("inbox", "archive"):
            f = self.dir(d) / Path(file).name
            if f.is_file():
                return f
        raise ValueError(f"patch {file} not found")

    def write(self, path, patch):
        """Rewrite a patch; once every entry is resolved it moves to archive/ (and back on revert)."""
        patch = {k: v for k, v in patch.items() if not k.startswith("_")}
        done = all(e.get("resolution") for e in patch["entries"])
        dest = self.dir("archive" if done else "inbox") / path.name
        dest.write_text(json.dumps(patch, ensure_ascii=False, indent=1), encoding="utf-8")
        if dest != path:
            path.unlink()
        return dest


def resolve(store, state, jpn, file, key, action, reviewer="", note=""):
    """Apply accept / reject / revert for one entry of an imported patch to `state` (mutated).
    The caller holds the state lock and writes state afterwards."""
    path = store.find(file)
    patch = json.loads(path.read_text(encoding="utf-8"))
    entry = next((e for e in patch["entries"] if e["key"] == key), None)
    if entry is None:
        raise ValueError(f"{key} is not in {file}")
    convs = state.setdefault("convs", {})
    res = entry.get("resolution")

    if action == "revert":  # undo an accept (restores the replaced conv) or reopen a rejected entry
        if not res:
            raise ValueError(f"{key} hasn't been reviewed yet")
        if res["action"] == "reject":
            entry.pop("resolution")
        elif content_hash(convs.get(key)) != entry["hash"]:
            raise ValueError(f"{key} has been edited since it was accepted; revert it by hand in the editor")
        else:
            if res.get("replaced") is None:
                convs.pop(key, None)
            else:
                convs[key] = res["replaced"]
            entry.pop("resolution")
    elif res:
        raise ValueError(f"{key} was already {res['action']}ed")
    elif action == "accept":
        a = assess(entry, convs.get(key), jpn.get(key))
        if a["problems"]:
            raise ValueError("; ".join(a["problems"]))
        conv = json.loads(json.dumps(entry["conv"]))
        conv["updated"] = now()
        conv["collab"] = {"hash": entry["hash"], "patch": patch["id"], "author": patch.get("author"),
                          "accepted": conv["updated"], "reviewer": reviewer, "note": note}
        entry["resolution"] = {"action": "accept", "at": conv["updated"], "reviewer": reviewer, "note": note,
                               "relation": a["relation"], "replaced": convs.get(key)}
        convs[key] = conv
    elif action == "reject":
        entry["resolution"] = {"action": "reject", "at": now(), "reviewer": reviewer, "note": note}
    else:
        raise ValueError(f"unknown action {action!r}")
    dest = store.write(path, patch)
    return {"ok": True, "file": dest.name, "where": dest.parent.name, "conv": convs.get(key)}
