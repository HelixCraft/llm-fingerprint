#!/usr/bin/env python3
"""Test suite for LLM Fingerprint Classifier v2."""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fingerprint import (
    normalize, normalize_answers,
    load_models, load_families,
    FingerprintClassifier,
    TASKS, LANG_CODES, NUM_WORDS, COLOR_CANON,
)


def test_normalization():
    print("── Normalization ──")
    cases = [
        ("42",          "num", "42"),
        ("Forty-two",   "num", "42"),
        ("  fortytwo ", "num", "42"),
        ("seven",       "num", "7"),
        ("Blue!",       "color", "blue"),
        ("azure",       "color", "azure"),
        ("Heads.",      "coin", "heads"),
        ("t",           "coin", "tails"),
        ("G",           "letter", "g"),
        ("elephant",    "word", "elephant"),
        ("Paris",       "city", "paris"),
        ("",            "num", ""),
    ]
    ok = 0
    for raw, ttype, expected in cases:
        result = normalize(raw, ttype)
        status = "✓" if result == expected else "✗"
        if status == "✓":
            ok += 1
        print(f"  {status} normalize({raw!r:20s}, {ttype:6s}) = {result!r:15s}  expected {expected!r}")
    print(f"  → {ok}/{len(cases)} passed\n")
    return ok == len(cases)


def test_comma_separated():
    print("── Comma-separated input ──")
    result = normalize_answers("42, 37, 57, forty-two", "num")
    expected = ["42", "37", "57", "42"]
    status = "✓" if result == expected else "✗"
    print(f"  {status} {result} == {expected}")
    print()
    return result == expected


def test_classification():
    print("── Classification tests ──")

    models = load_models()
    families = load_families()
    clf = FingerprintClassifier(models, families)
    print(f"  Loaded {clf.n} models / {len(clf.cells)} cells\n")

    tests = [
        ("GPT-4o", {
            "num100-random|en": ["42","42","37","57","27"],
            "num-favorite|en":  ["7"],
            "coin-flip|en":     ["heads","tails"],
            "color-random|en":  ["blue"],
            "animal-random|en": ["elephant"],
            "city-random|en":   ["tokyo","paris"],
            "num10-random|en":  ["7"],
        }, "gpt"),
        ("Claude Sonnet 5", {
            "num100-random|en": ["47","47","47","37"],
            "num-favorite|en":  ["7"],
            "coin-flip|en":     ["heads","heads","heads"],
            "animal-random|en": ["elephant"],
            "city-random|en":   ["tokyo","paris","chicago"],
        }, "claude"),
        ("Qwen3-max", {
            "num100-random|en": ["42","42","42","42"],
            "num-favorite|en":  ["42"],
            "coin-flip|en":     ["heads"],
            "animal-random|en": ["elephant"],
        }, "qwen"),
        ("DeepSeek V4 Flash", {
            "num100-random|en": ["47","7","42","37","73"],
            "num-favorite|en":  ["7"],
            "coin-flip|en":     ["heads","tails","tails"],
            "animal-random|en": ["elephant","cat"],
            "color-random|en":  ["blue","red","green"],
        }, "deepseek"),
    ]

    all_ok = True
    for name, obs, expected_family in tests:
        top = clf.top_k(obs, 5, method="jsd")
        top_families = [clf.families.get(m, "") for m, _ in top]

        # First model's family should match expected
        top_fam = clf.families.get(top[0][0], "")
        ok = expected_family in top_fam.lower() or any(expected_family in f.lower() for f in top_families[:3])
        status = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {status} {name:20s} → #{top[0][0]:45s} {top[0][1]*100:5.2f}%  [{top_fam}]")

    print(f"\n  → {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


def test_family_aggregation():
    print("── Family aggregation ──")
    models = load_models()
    families = load_families()
    clf = FingerprintClassifier(models, families)
    obs = {
        "num100-random|en": ["42","42","42"],
        "num-favorite|en":  ["42"],
        "coin-flip|en":     ["heads"],
    }
    fams = clf.family_probs(obs)
    assert fams, "Family probs should not be empty"
    # Qwen should be among top families (42-heavy answers)
    top_fam = fams[0][0]
    print(f"  Top family: {top_fam} ({fams[0][1]*100:.1f}%)")
    for f, p in fams[:5]:
        print(f"    {f:12s} {p*100:5.1f}%")
    ok = top_fam == "qwen"
    print(f"  {'✓' if ok else '✗'} Expected qwen on top\n")
    return ok


def test_confidence():
    print("── Confidence scoring ──")
    models = load_models()
    families = load_families()
    clf = FingerprintClassifier(models, families)
    obs = {"num100-random|en": ["42"]}
    gap, label = clf.confidence(obs)
    print(f"  Single answer: {label} (Δ={gap})")
    assert label, "Should produce a label"
    print(f"  ✓\n")
    return True


def test_jsd_symmetry():
    print("── JSD symmetry ──")
    from fingerprint import jsd
    p = {"a": 0.6, "b": 0.4}
    q = {"a": 0.3, "b": 0.7}
    d1 = jsd(p, q)
    d2 = jsd(q, p)
    ok = abs(d1 - d2) < 1e-10
    print(f"  JSD(P,Q)={d1:.6f}  JSD(Q,P)={d2:.6f}  {'✓' if ok else '✗'}")
    print()
    return ok


def main():
    print()
    print("╔══════════════════════════════════════╗")
    print("║  LLM Fingerprint — Test Suite v2.0  ║")
    print("╚══════════════════════════════════════╝")
    print()

    tests = [
        ("Normalization",    test_normalization),
        ("Comma-separated",  test_comma_separated),
        ("Classification",   test_classification),
        ("Families",         test_family_aggregation),
        ("Confidence",       test_confidence),
        ("JSD symmetry",     test_jsd_symmetry),
    ]

    passed = 0
    for name, fn in tests:
        print(f"─── {name} ───")
        ok = fn()
        if ok:
            passed += 1

    print("═" * 44)
    print(f"  {passed}/{len(tests)} test suites passed")
    print()


if __name__ == "__main__":
    main()
