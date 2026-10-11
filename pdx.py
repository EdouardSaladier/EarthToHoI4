"""Minimal Paradox script (Clausewitz) reader / writer for the HOI4 mod export.

parse(text) -> list of (key, op, value) entries; a value is a string (bare
words keep their text, quoted strings keep their quotes) or a list of entries
(a block). Bare values inside a block (`provinces = { 1 2 3 }`) are entries
with key None. Typed blocks (`color = rgb { 1 2 3 }`) give the value
("rgb", [entries]) as a tuple. Comments are dropped.

dump(entries) -> text, tab-indented. Round trip: parse(dump(parse(t))) ==
parse(t)."""
import re

_TOKEN = re.compile(r'''
    (?P<ws>\s+)
  | (?P<comment>\#[^\n]*)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<op><=|>=|!=|==|\?=|=|<|>)
  | (?P<open>\{)
  | (?P<close>\})
  | (?P<word>[^\s{}=<>!"#?]+(?:\?[^\s{}=<>!"#]*)?)
''', re.X)

TYPED = {"rgb", "hsv", "hsv360", "RGB", "HSV"}


class ParseError(ValueError):
    pass


def tokenize(text):
    out = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m:
            # a stray character (e.g. a lone '"' or '!'): keep it as a word
            out.append(("word", text[pos]))
            pos += 1
            continue
        pos = m.end()
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            continue
        out.append((kind, m.group(kind)))
    return out


def parse(text):
    if text.startswith("﻿"):
        text = text[1:]
    toks = tokenize(text)
    i = 0

    def block(top):
        nonlocal i
        entries = []
        while i < len(toks):
            kind, val = toks[i]
            if kind == "close":
                if top:  # unbalanced: skip it (the game tolerates some)
                    i += 1
                    continue
                i += 1
                return entries
            if kind == "open":  # anonymous block as a bare value
                i += 1
                entries.append((None, None, block(False)))
                continue
            if kind == "op":  # stray operator
                i += 1
                continue
            # word / str: key (if an operator follows) or bare value
            if i + 1 < len(toks) and toks[i + 1][0] == "op":
                key, op = val, toks[i + 1][1]
                i += 2
                if i >= len(toks):
                    entries.append((key, op, ""))
                    break
                k2, v2 = toks[i]
                if k2 == "open":
                    i += 1
                    entries.append((key, op, block(False)))
                elif k2 in ("word", "str"):
                    i += 1
                    if k2 == "word" and v2 in TYPED and i < len(toks) and toks[i][0] == "open":
                        i += 1
                        entries.append((key, op, (v2, block(False))))
                    else:
                        entries.append((key, op, v2))
                else:  # "key = }" etc.
                    entries.append((key, op, ""))
            else:
                i += 1
                entries.append((None, None, val))
        if not top:
            pass  # unclosed block at end of file: accept
        return entries

    return block(True)


def _dump_value(v, depth):
    if isinstance(v, tuple):
        return v[0] + " " + _dump_block(v[1], depth)
    if isinstance(v, list):
        return _dump_block(v, depth)
    return v


def _dump_block(entries, depth):
    if not entries:
        return "{ }"
    if all(k is None and isinstance(v, str) for k, _o, v in entries) and len(entries) <= 40:
        return "{ " + " ".join(v for _k, _o, v in entries) + " }"
    ind = "\t" * (depth + 1)
    lines = ["{"]
    for k, o, v in entries:
        lines.append(ind + _dump_entry(k, o, v, depth + 1))
    lines.append("\t" * depth + "}")
    return "\n".join(lines)


def _dump_entry(k, o, v, depth):
    if k is None:
        return _dump_value(v, depth)
    return f"{k} {o} {_dump_value(v, depth)}"


def dump(entries, depth=0):
    return "\n".join(("\t" * depth) + _dump_entry(k, o, v, depth) for k, o, v in entries) + "\n"


def get(entries, key, default=None):
    """First value of `key` (case-insensitive) in a block."""
    kl = key.lower()
    for k, _o, v in entries:
        if k is not None and k.lower() == kl:
            return v
    return default


def get_all(entries, key):
    kl = key.lower()
    return [v for k, _o, v in entries if k is not None and k.lower() == kl]


def walk(entries):
    """Every (key, op, value) in the tree, depth first."""
    for e in entries:
        yield e
        v = e[2]
        if isinstance(v, tuple):
            v = v[1]
        if isinstance(v, list):
            yield from walk(v)


def read(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return parse(f.read())
