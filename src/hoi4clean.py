"""Dead-reference cleaner for the base game's script files (HOI4 mod export).

The mod loads none of the base game's countries (its 00_countries.txt is
replaced), and its country focus trees, decisions and events are unloaded
(hoi4mod.REPLACE_PATHS). Everything else of the base game still loads and
names those things -- 47k error.log lines, mostly "X - is not in the tag
list", "Invalid focus scripted in trigger", "Invalid Decision ID". This
module rewrites each affected file with the dead references taken out:

* a *dead token* is a removed country tag, focus, decision or event id;
* in triggers (limit, allowed, available, visible, potential, trigger, ...),
  a test naming a dead token is false: removed from OR / NOT / NOR (it
  changes nothing there), it makes an AND-type block false, and a false
  gate drops the definition it gates (an idea, a MIO, a name group, an AI
  plan, an if / random_list branch ...);
* in effects, a scope on a dead country (``GER = { ... }``) and an effect
  with a dead token among its parameters are removed;
* lists of tags (``for_countries = { GER ITA }``) lose their dead members;
  an emptied list makes its definition false.

clean_file(entries, dead) -> (entries, changed). The caller writes changed
files under the same name (overriding the base game's)."""
import pdx

TRIGGER_KEYS = {"limit", "trigger", "allowed", "allow", "available", "visible", "potential", "is_valid",
                "can_be_used", "can_use", "is_good", "activation", "allowed_civil_war", "target_trigger",
                "target_root_trigger", "visible_trigger", "ai_will_do_trigger", "requirements", "enable",
                "available_if_capitulated", "custom_trigger_tooltip", "is_shown", "unlock_trigger",
                "on_trigger", "show_condition", "valid", "active", "triggered", "ready", "condition"}
# triggers that make something stop / go away: always false -> "always = no", the
# definition stays
NEG_TRIGGER_KEYS = {"cancel", "cancel_trigger", "remove_trigger", "allowed_to_remove", "bypass",
                    "cancel_if_invalid_trigger", "can_be_cancelled", "remove_if", "fail_trigger",
                    "complete_trigger", "timeout_trigger"}
OR_KEYS = {"or"}
NOT_KEYS = {"not", "nor"}
AND_KEYS = {"and"}
NAND_KEYS = {"nand"}


def _lk(k):
    """Keys compared case-insensitively (the game accepts IF, ELSE_IF, Or ...)."""
    return k.lower() if isinstance(k, str) else k
# modifier-like blocks: a false test inside means the block never applies -> drop it
GATED_BLOCKS = {"modifier", "mult_modifier", "add_modifier", "ai_will_do_modifier"}
# tag lists that exclude or prefer: emptied, only the list goes (the generic AI
# templates' blocked_for -- dropping them left 4 of 11 roles: the wiki's mid-game
# "division template with no ai_templates entry" crash)
LIST_DROP_ONLY = {"blocked_for", "preferred_countries", "enable_for_controllers"}
# blocks that can't be empty: emptied by the cleaning -> their definition goes
REQUIRED_BLOCKS = {"ratio_progress"}


class _Dead(Exception):
    """Raised inside a trigger evaluation: the block is always false."""


QUOTED_TAG_KEYS = {"id", "tag", "original_tag", "target", "country", "creator"}


def _is_dead(v, dead, key=None):
    """A dead token; quoted only under keys that hold tags (id = "YUG" --
    abbreviation = "PAR" is a name, not Paraguay)."""
    if not isinstance(v, str):
        return False
    if v in dead:
        return True
    if ":" in v or "." in v or "|" in v:  # mio:CZE_..., constant:country_groups.nordics, idea_desc|CZE_x
        parts = v.strip('"').replace(":", ".").replace("|", ".").split(".")
        if any(p in dead for p in parts if p):  # incl. the first: SWI.num_owned_states
            return True
    return (key is not None and key.lower() in QUOTED_TAG_KEYS and len(v) > 2 and v[0] == '"' and v[-1] == '"'
            and v[1:-1] in dead)


def _trigger_if(k, v, dead):
    """A trigger-context if / else_if / else: (entry or None, limit_false).
    A limit that can't hold means the branch is skipped (true, not false);
    a body that can't hold becomes always = no."""
    lim = pdx.get(v, "limit") if _lk(k) != "else" else None
    if lim is not None:
        lim_new, lim_false = _clean_trigger(lim, dead, "AND")
        if lim_false:
            return None, True
    body = [e for e in v if _lk(e[0]) != "limit"]
    body_new, body_false = _clean_trigger(body, dead, "AND")
    if body_false:
        body_new = [("always", "=", "no")]
    new = ([("limit", "=", lim_new)] if lim is not None else []) + body_new
    return (k, "=", new), False


def _clean_trigger(entries, dead, mode="AND"):
    """(entries, false?) of a trigger block. mode: AND (all must hold), OR,
    NOT (none may hold)."""
    out = []
    chain_dropped = chain_open = False
    for k, o, v in entries:
        kl = _lk(k)
        if kl in ("if", "else_if", "else") and isinstance(v, list):
            k = kl
            if k in ("else_if", "else") and not (chain_dropped and chain_open):
                if not chain_open:  # an else of a kept branch
                    e, _f = _trigger_if(k, v, dead)
                    if e:
                        out.append(e)
                    continue
            if k == "else" and chain_dropped and chain_open:
                body_new, body_false = _clean_trigger(v, dead, "AND")
                if body_false:
                    if mode == "AND":
                        return [], True
                else:
                    out.extend(body_new)
                chain_dropped = chain_open = False
                continue
            kk = "if" if (k == "else_if" and chain_dropped and chain_open) else k
            if kk == "if":
                chain_open = True
            e, lim_false = _trigger_if(kk, v, dead)
            if lim_false:
                chain_dropped = True
                continue
            out.append(e)
            chain_dropped = False
            chain_open = False
            continue
        chain_dropped = chain_open = False
        val_false = False
        if k is None:
            if _is_dead(v, dead):
                val_false = True
            else:
                out.append((k, o, v))
                continue
        elif k in dead:  # scope on a dead country
            val_false = True
        elif isinstance(v, str):
            val_false = _is_dead(v, dead, k)
            if not val_false:
                out.append((k, o, v))
                continue
        elif isinstance(v, tuple):
            out.append((k, o, v))
            continue
        else:  # block
            if kl in OR_KEYS:
                sub, f = _clean_trigger(v, dead, "OR")
            elif kl in NOT_KEYS:
                sub, f = _clean_trigger(v, dead, "NOT")
                if not sub:  # NOT of nothing (all false members): true -> drop it
                    continue
            elif kl in NAND_KEYS:
                sub, f = _clean_trigger(v, dead, "AND")
                if f:  # NAND of a false AND: true
                    continue
            else:  # AND, scopes, iterators (any_country = { ... }), count_triggers, if ...
                sub, f = _clean_trigger(v, dead, "AND")
            if f:
                val_false = True
            else:
                out.append((k, o, sub))
                continue
        if val_false:
            if mode == "AND":
                return [], True
            # OR: a false member changes nothing; NOT: likewise
            continue
    if mode == "OR" and not out and entries:
        return [], True
    return out, False


def _clean_effect(entries, dead):
    """(entries, broken?) of an effect / definition block: broken = it had a
    dead token among its own parameters (the caller drops it). An if whose
    limit can't hold is dropped: a following else_if becomes the if, a
    following else's contents run as they are."""
    out = []
    chain_dropped = False  # the last if / else_if of the chain was dropped
    chain_open = False     # ... and nothing of the chain was kept yet
    for k, o, v in entries:
        if _lk(k) in ("if", "else_if", "else"):
            k = _lk(k)
        if k in ("else_if", "else") and chain_dropped and chain_open:
            if k == "else":
                sub, broken = _clean_effect(v, dead) if isinstance(v, list) else ([], True)
                if not broken:
                    out.extend(sub)
                chain_dropped = chain_open = False
                continue
            k = "if"  # the chain's first kept branch
        if k == "if":
            chain_open = True
        elif k not in ("else_if", "else"):
            chain_open = False
        chain_dropped = False
        n_before = len(out)
        _clean_effect_entry(k, o, v, dead, out)
        if k in ("if", "else_if") and len(out) == n_before:
            chain_dropped = True
        elif k in ("if", "else_if", "else"):
            chain_open = False
    if any(e is _BROKEN for e in out):
        return [], True
    return out, False


_BROKEN = ("__broken__", None, None)


def _clean_effect_entry(k, o, v, dead, out):
    """Appends the cleaned entry to out (nothing if removed; _BROKEN if the
    enclosing block must go)."""
    if True:
        if k is None:
            if not _is_dead(v, dead):  # else: a dead member of a list
                out.append((k, o, v))
            return
        if k in dead or ((":" in k or "." in k or "|" in k) and _is_dead(k, dead)):
            return  # scope on a dead country / mio:... / character:...
        if isinstance(v, str):
            out.append(_BROKEN if _is_dead(v, dead, k) else (k, o, v))
            return
        if isinstance(v, tuple):
            out.append((k, o, v))
            return
        kl = _lk(k)
        if kl in NEG_TRIGGER_KEYS:
            sub, f = _clean_trigger(v, dead, "AND")
            out.append((k, o, [("always", "=", "no")] if f else sub))
            return
        if kl in TRIGGER_KEYS or kl in OR_KEYS or kl in NOT_KEYS or kl in AND_KEYS or kl in NAND_KEYS:
            mode = "OR" if kl in OR_KEYS else "NOT" if kl in NOT_KEYS else "AND"
            sub, f = _clean_trigger(v, dead, mode)
            if f and _KEEP_GATED[0]:
                out.append((k, o, [("always", "=", "no")]))
            elif f:
                out.append(_BROKEN)  # a false gate: the whole block never applies
            elif not (kl in NOT_KEYS and not sub):
                if not sub and v and kl in TRIGGER_KEYS:
                    sub = [("always", "=", "yes")]  # emptied (was "NOT a removed country"): holds;
                    # the game: "allowed trigger is empty", "missing or empty enable trigger"
                out.append((k, o, sub))
            return
        had_list = bool(v) and all(e[0] is None for e in v)
        sub, broken = _clean_effect(v, dead)
        if broken:
            if kl in REQUIRED_BLOCKS:
                out.append(_BROKEN)  # a faction manifest whose ratio_progress named a removed constant
            return  # the effect / definition block that held a dead parameter
        if had_list and not sub:
            if kl in LIST_DROP_ONLY:
                return  # blocked_for = { GER ENG } emptied: nobody excluded -- only the list goes
            out.append(_BROKEN)  # e.g. for_countries = { GER } emptied: the definition can't apply
            return
        if kl in GATED_BLOCKS and not sub:
            return
        if v and not sub and all(e[0] is not None and isinstance(e[2], str) for e in v):
            return  # a parameter block emptied (add_to_array = { BRA.x = THIS }): meaningless
        if kl in REQUIRED_BLOCKS and not sub and v:
            out.append(_BROKEN)  # e.g. a faction manifest whose ratio_progress was all removed
            return
        out.append((k, o, sub))




_KEEP_GATED = [False]


def clean_file(entries, dead, keep_gated=False):
    """A file's entries with dead references removed; (entries, changed).
    keep_gated: a definition whose gate can't hold stays, gated by always = no
    (lists whose length the game checks, like army_icons.txt)."""
    _KEEP_GATED[0] = keep_gated
    try:
        new, _broken = _clean_effect(entries, dead)
    finally:
        _KEEP_GATED[0] = False
    return new, new != entries


def dead_tokens(van, kept_focus, kept_decisions, kept_events, our_tags):
    """Removed tags, focuses, decisions and events of the base game."""
    import re
    dead = {t for t in van.tags if t not in set(van.dynamic_tags) and t not in our_tags}
    focus, decisions, events = set(), set(), set()
    for f in (van.game / "common/national_focus").glob("*.txt"):
        for k, _o, v in pdx.walk(pdx.read(f)):
            if k in ("focus", "shared_focus") and isinstance(v, list):
                i = pdx.get(v, "id")
                if i:
                    focus.add(i)
    for f in (van.game / "common/decisions").glob("*.txt"):
        for k, _o, v in pdx.read(f):
            if isinstance(v, list):
                decisions.update(dk for dk, _do, dv in v if dk and isinstance(dv, list))
    for f in (van.game / "events").glob("*.txt"):
        for k, _o, v in pdx.read(f):
            if isinstance(v, list) and k and k.endswith("_event"):
                i = pdx.get(v, "id")
                if i:
                    events.add(i)
    dead |= focus - set(kept_focus)
    dead |= decisions - set(kept_decisions)
    dead |= events - set(kept_events)
    return {d for d in dead if d and not re.fullmatch(r"\d+|yes|no", d)}
