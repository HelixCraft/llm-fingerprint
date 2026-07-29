#!/usr/bin/env python3
"""
LLM Fingerprint Classifier — identify which model answered your questions.

Based on:  Bruckner (2026) "One Token Is Enough" (arXiv:2607.10252)
Dataset:   Zenodo DOI:10.5281/zenodo.21278557
Method:    Jensen-Shannon Divergence (paper §IV-C) + Naive Bayes
Models:    170 models · 10 tasks · 4 languages (en/ru/zh/ar)
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fingerprint import (
    FingerprintClassifier as _C,
    load_models, load_families,
    normalize_answers,
    TASKS, LANG_CODES,
)

VERSION = "2.0.0"

CELLS = [f"{tid}|lang" for tid, _, _ in TASKS]
TASK_MAP = {tid: (label, ttype) for tid, label, ttype in TASKS}
LABEL_TO_TID = {label: tid for tid, label, _ in TASKS}
OUT_FILE = os.path.join(_HERE, "last_result.json")


def banner():
    print()
    print("╔" + "═" * 56 + "╗")
    print(f"║  LLM Fingerprint Classifier v{VERSION}              ║")
    print(f"║  Bruckner (2026)  One Token Is Enough        ║")
    print(f"║  {170} models · {len(TASKS)} tasks · {len(LANG_CODES)} languages            ║")
    print("╚" + "═" * 56 + "╝")
    print()


def pick_language() -> str:
    while True:
        inp = input("  Language [en/ru/zh/ar] (enter=en): ").strip().lower()
        if not inp:
            return "en"
        if inp in LANG_CODES:
            return inp
        print(f"  Choose: {', '.join(LANG_CODES)}")


def collect(lang: str) -> dict:
    print(f"\n  Language: {LANG_CODES[lang]}\n")
    obs = {}

    for tid, label, ttype in TASKS:
        cell = f"{tid}|{lang}"
        inp = input(f"  {label:30s} [{lang}]: ").strip()
        if not inp:
            print("    · skip")
            continue
        answers = normalize_answers(inp, ttype)
        if not answers:
            print("    · invalid (ignored)")
            continue
        obs[cell] = answers
        print(f"    → {answers}")

    return obs


def display(clf: _C, obs: dict, method: str):
    print()
    print("╔" + "═" * 56 + "╗")
    lbl = "JSD  (paper §IV-C)" if method == "jsd" else "Naive Bayes"
    print(f"║  PREDICTION  —  {lbl:30s}║")
    print("╚" + "═" * 56 + "╝")

    top = clf.top_k(obs, 15, method=method)
    if not top:
        print("  No data.\n")
        return

    print()
    for i, (m, p) in enumerate(top, 1):
        pct = p * 100
        star = "★" if i == 1 else " "
        fam = clf.families.get(m, "?")
        print(f"  {star} {i:2d}. {m:45s} {pct:5.2f}%  [{fam}]")

    print()
    print("  Model families:")
    for fam, p in clf.family_probs(obs, method=method):
        pct = p * 100
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        print(f"    {fam:12s} {bar} {pct:5.1f}%")

    print()
    ev = clf.evidence(obs, method=method)
    print(f"  Best:  {ev['best']}  ({ev['best_prob']*100:.2f}%)")
    if ev["second"]:
        print(f"  Next:  {ev['second']}  ({ev['second_prob']*100:.2f}%)")

    if ev["for"]:
        print("\n  Signals FOR:")
        for diff, cell, answers in ev["for"][:5]:
            tid, lg = cell.split("|")
            lbl, _ = TASK_MAP.get(tid, (tid, ""))
            ln = LANG_CODES.get(lg, lg)
            print(f"    + {lbl} ({ln}): {answers}")

    if ev["against"]:
        print("\n  Signals AGAINST:")
        for diff, cell, answers in ev["against"][:3]:
            tid, lg = cell.split("|")
            lbl, _ = TASK_MAP.get(tid, (tid, ""))
            ln = LANG_CODES.get(lg, lg)
            print(f"    - {lbl} ({ln}): {answers}")

    gap, conf = clf.confidence(obs, method=method)
    ent = clf.entropy_bits(obs, method=method)
    print(f"\n  Confidence: {conf}  (Δ={gap})")
    print(f"  Entropy:    {ent} bits")
    print(f"  Cells:      {len(obs)}/{len(TASKS)}")
    print("═" * 56)
    print()


def save(obs: dict, clf: _C, method: str):
    top = clf.top_k(obs, 50, method=method)
    probs = clf.proba(obs, method=method)
    jsd_scores = {m: round(clf.score_jsd(m, obs), 4) for m in clf.models}
    families = clf.family_probs(obs, method=method)

    out = {
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "method": method,
        "language": next((lg for cell in obs for lg in LANG_CODES if cell.endswith(f"|{lg}")), "?"),
        "input": {c: a for c, a in obs.items()},
        "predictions": [{"model": m, "probability": round(p, 6), "family": clf.families.get(m, "other"), "jsd": jsd_scores.get(m)} for m, p in top],
        "families": {f: round(p, 6) for f, p in families},
        "confidence": clf.confidence(obs, method=method)[1],
        "entropy": clf.entropy_bits(obs, method=method),
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return OUT_FILE


def load_yaml(path: str) -> dict:
    """Load answers from a YAML file.

    Format:
        language: en
        answers:
          Random number (1-100): 42, 37, 57
          Favorite number: 7
          Coin flip: heads, tails
          ...
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print(f"  Error: {path} must contain a YAML mapping.")
        sys.exit(1)

    lang = str(data.get("language", "en"))
    if lang not in LANG_CODES:
        print(f"  Error: unknown language '{lang}'. Choose: {', '.join(LANG_CODES)}")
        sys.exit(1)

    raw_answers = data.get("answers", {})
    if not isinstance(raw_answers, dict) or not raw_answers:
        print(f"  Error: {path} must contain an 'answers' mapping.")
        sys.exit(1)

    obs = {}
    for key, value in raw_answers.items():
        key = key.strip()
        tid = LABEL_TO_TID.get(key, key)
        _, ttype = TASK_MAP.get(tid, (None, None))
        if ttype is None:
            print(f"  Warning: unknown question '{key}', skipping.")
            continue
        cell = f"{tid}|{lang}"

        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
        elif isinstance(value, (int, float)):
            parts = [str(value)]
        elif isinstance(value, list):
            parts = [str(v).strip() for v in value if str(v).strip()]
        else:
            print(f"  Warning: invalid value for '{key}', skipping.")
            continue

        answers = normalize_answers(",".join(parts), ttype)
        if answers:
            obs[cell] = answers
            print(f"  Loaded: {key:30s} → {answers}")
        else:
            print(f"  Warning: no valid answers for '{key}', skipping.")

    if not obs:
        print("  Error: no valid answers found in YAML file.")
        sys.exit(1)

    return obs, lang


def main():
    banner()

    print("  Loading fingerprints … ", end="", flush=True)
    models = load_models()
    families = load_families()
    clf = _C(models, families)
    print(f"{clf.n} models, {len(clf.cells)} cells.\n")

    method = "jsd"

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"  Loading answers from: {path}\n")
        obs, lang = load_yaml(path)
        print(f"\n  Language: {LANG_CODES[lang]}\n")
        display(clf, obs, method=method)
        path = save(obs, clf, method=method)
        print(f"  Results saved: {path}\n")
        print("  Done.\n")
        return

    while True:
        lang = pick_language()
        obs = collect(lang)
        if not obs:
            print("  Nothing entered.\n")
            break

        display(clf, obs, method=method)
        path = save(obs, clf, method=method)
        print(f"  Results saved: {path}\n")

        cmd = input("  Another round? [Y/n]: ").strip().lower()
        if cmd and cmd not in ("y", "yes", ""):
            break

    print("  Done.\n")


if __name__ == "__main__":
    main()
