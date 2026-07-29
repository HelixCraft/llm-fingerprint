"""
LLM Fingerprint Classifier

Based on:  Bruckner (2026) "One Token Is Enough" (arXiv:2607.10252)
Dataset:   Zenodo DOI:10.5281/zenodo.21278557
Method:    Jensen-Shannon Divergence + Naive Bayes

170 models · 10 tasks · 4 languages (en/ru/zh/ar)
"""

import json
import math
import os
import unicodedata
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_PATH = os.path.join(DATA_DIR, "models.json")

# ─── Task definitions ────────────────────────────────────────────────

TASKS = [
    ("num100-random", "Random number (1-100)",  "num"),
    ("num10-random",  "Random number (1-10)",   "num"),
    ("num-favorite",  "Favorite number",        "num"),
    ("letter-random", "Random letter",          "letter"),
    ("word-random",   "Random word",            "word"),
    ("color-random",  "Random color",           "color"),
    ("color-favorite","Favorite color",         "color"),
    ("animal-random", "Random animal",          "animal"),
    ("city-random",   "Random city",            "city"),
    ("coin-flip",     "Coin flip",              "coin"),
]

LANG_CODES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ar": "Arabic"}

# ─── Color map (paper §IV-B) ─────────────────────────────────────────

COLOR_CANON = {
    "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
    "orange": "orange", "purple": "purple", "pink": "pink", "black": "black",
    "white": "white", "brown": "brown", "gray": "gray", "grey": "gray",
    "cyan": "cyan", "teal": "teal", "turquoise": "turquoise", "azure": "azure",
    "cerulean": "cerulean", "indigo": "indigo", "violet": "violet",
    "magenta": "magenta", "crimson": "crimson", "scarlet": "scarlet",
    "gold": "gold", "silver": "silver", "navy": "navy", "maroon": "maroon",
    "olive": "olive", "lime": "lime", "aqua": "aqua", "coral": "coral",
    "salmon": "salmon", "khaki": "khaki", "lavender": "lavender",
    "beige": "beige", "ivory": "ivory", "tan": "tan", "peach": "peach",
    "mint": "mint", "emerald": "emerald", "ruby": "ruby", "sapphire": "sapphire",
    "jade": "jade", "amber": "amber", "burgundy": "burgundy",
}

# ─── Number words (from the paper's multilingual normalization) ──────

NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
    "fortytwo": "42", "fortyone": "41", "fortythree": "43",
    "fortyfour": "44", "fortyfive": "45", "fortysix": "46",
    "fortyseven": "47", "fortyeight": "48", "fortynine": "49",
    "fiftyone": "51", "fiftytwo": "52", "fiftythree": "53",
    "fiftyfour": "54", "fiftyfive": "55", "fiftysix": "56",
    "fiftyseven": "57", "fiftyeight": "58", "fiftynine": "59",
    "sixtyone": "61", "sixtytwo": "62", "sixtythree": "63",
    "sixtyfour": "64", "sixtyfive": "65", "sixtysix": "66",
    "sixtyseven": "67", "sixtyeight": "68", "sixtynine": "69",
    "seventyone": "71", "seventytwo": "72", "seventythree": "73",
    "seventyfour": "74", "seventyfive": "75", "seventysix": "76",
    "seventyseven": "77", "seventyeight": "78", "seventynine": "79",
    "eightyone": "81", "eightytwo": "82", "eightythree": "83",
    "eightyfour": "84", "eightyfive": "85", "eightysix": "86",
    "eightyseven": "87", "eightyeight": "88", "eightynine": "89",
    "ninetyone": "91", "ninetytwo": "92", "ninetythree": "93",
    "ninetyfour": "94", "ninetyfive": "95", "ninetysix": "96",
    "ninetyseven": "97", "ninetyeight": "98", "ninetynine": "99",
}


# ═══════════════════════════════════════════════════════════════════════
# NORMALISATION (paper §IV-B)
# ═══════════════════════════════════════════════════════════════════════

def normalize(raw: str, task_type: str) -> str:
    """Normalize a raw answer per the paper's protocol.

    Steps: 1. Unicode NFC  2. Punctuation stripping
           3. Case folding  4. First-token extraction
           5. Task-specific canonicalization (digits, colours, coin)
    """
    raw = raw.strip()
    if not raw:
        return ""

    s = unicodedata.normalize("NFC", raw)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P")
                and ch not in '""\'\u201c\u201d\u2018\u2019')
    s = s.strip().lower()
    s = s.split()[0] if s.split() else ""
    if not s:
        return ""

    if task_type == "num":
        s = NUM_WORDS.get(s, s)
        s = "".join(c for c in s if c.isdigit() or c == "-")
        s = s.lstrip("0") or "0"
        try:
            v = int(s)
            if v < 1: return "1"
            if v > 100: return "100"
            return str(v)
        except ValueError:
            return ""

    if task_type == "color":
        return COLOR_CANON.get(s, s)

    if task_type == "coin":
        if s in ("heads", "head", "h"):
            return "heads"
        if s in ("tails", "tail", "t"):
            return "tails"
        return ""

    if task_type == "letter":
        s = "".join(c for c in s if c.isalpha())
        return s[:1] if s else ""

    return s


def normalize_answers(raw_input: str, task_type: str) -> list[str]:
    """Split comma-separated input, normalize each part."""
    return [normalize(p.strip(), task_type)
            for p in raw_input.split(",")
            if normalize(p.strip(), task_type)]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_models() -> dict:
    """Load precomputed fingerprint distributions.

    Returns {model_id: {cell_key: {answer: probability}}}
    """
    with open(MODELS_PATH) as f:
        data = json.load(f)
    return data["models"]


def load_families() -> dict:
    """Load model → family mapping."""
    with open(MODELS_PATH) as f:
        data = json.load(f)
    return data.get("families", {})


# ═══════════════════════════════════════════════════════════════════════
# JENSEN–SHANNON DIVERGENCE (paper §IV-C)
# ═══════════════════════════════════════════════════════════════════════

def kl(p: dict, q: dict) -> float:
    """KL(P‖Q) base 2."""
    d = 0.0
    for a, pa in p.items():
        qa = q.get(a, 1e-12)
        if pa > 0:
            d += pa * math.log2(pa / qa)
    return d


def jsd(p: dict, q: dict) -> float:
    """Jensen–Shannon divergence base 2."""
    keys = set(p) | set(q)
    m = {a: (p.get(a, 0.0) + q.get(a, 0.0)) / 2.0 for a in keys}
    return (kl(p, m) + kl(q, m)) / 2.0


def empirical(answers: list) -> dict:
    """Build empirical distribution from answer list."""
    n = len(answers)
    if n == 0:
        return {}
    d = {}
    for a in answers:
        d[a] = d.get(a, 0) + 1.0 / n
    return d


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

class FingerprintClassifier:
    """Naive Bayes + JSD classifier over 170 model fingerprints."""

    def __init__(self, models: dict, families: dict):
        self.models = models
        self.families = families
        self.n = len(models)
        self.prior = math.log(1.0 / self.n)
        self.cells = sorted(set(c for fp in models.values() for c in fp))

    # ── JSD score (paper's primary method) ──────────────────────

    def score_jsd(self, model: str, obs: dict) -> float:
        scores = []
        for cell, answers in obs.items():
            emp = empirical(answers)
            md = self.models.get(model, {}).get(cell)
            if md and emp:
                scores.append(jsd(emp, md))
        return sum(scores) / len(scores) if scores else float("inf")

    # ── Naive Bayes log-likelihood ──────────────────────────────

    def score_nb(self, model: str, obs: dict, alpha: float = 0.1) -> float:
        logp = self.prior
        for cell, answers in obs.items():
            dist = self.models.get(model, {}).get(cell, {})
            if not dist:
                continue
            total = sum(dist.values())
            if total == 0:
                continue
            vocab = len(dist)
            for ans in answers:
                p = dist.get(ans, 0.0)
                if p > 0:
                    logp += math.log(p + 1e-12)
                else:
                    logp += math.log(alpha / (total + alpha * (vocab + 1)))
        return logp

    # ── Probability output ──────────────────────────────────────

    def proba(self, obs: dict, method: str = "jsd", T: float = 0.15) -> dict:
        if method == "nb":
            scores = {m: self.score_nb(m, obs) for m in self.models}
            max_s = max(scores.values())
            exp_s = {m: math.exp(s - max_s) for m, s in scores.items()}
        else:
            scores = {m: self.score_jsd(m, obs) for m in self.models}
            min_s = min(scores.values())
            shifted = {m: -(s - min_s) / T for m, s in scores.items()}
            max_sh = max(shifted.values())
            exp_s = {m: math.exp(s - max_sh) for m, s in shifted.items()}
        total = sum(exp_s.values()) or 1.0
        return {m: exp_s[m] / total for m in scores}

    def top_k(self, obs: dict, k: int = 20, method: str = "jsd") -> list:
        probs = self.proba(obs, method=method)
        return sorted(probs.items(), key=lambda x: -x[1])[:k]

    def family_probs(self, obs: dict, method: str = "jsd") -> list:
        probs = self.proba(obs, method=method)
        fam = defaultdict(float)
        for m, p in probs.items():
            fam[self.families.get(m, "other")] += p
        return sorted(fam.items(), key=lambda x: -x[1])

    def evidence(self, obs: dict, method: str = "jsd") -> dict:
        probs = self.proba(obs, method=method)
        sm = sorted(probs, key=probs.get, reverse=True)
        best, bp = sm[0], probs[sm[0]]
        second, sp = (sm[1], probs[sm[1]]) if len(sm) > 1 else ("", 0)

        ev = []
        for cell, answers in obs.items():
            if method == "jsd":
                s1 = -self.score_jsd(best, {cell: answers})
                s2 = -self.score_jsd(second, {cell: answers}) if second else -100
            else:
                s1 = self.score_nb(best, {cell: answers})
                s2 = self.score_nb(second, {cell: answers}) if second else -100
            ev.append((s1 - s2, cell, answers))

        ev.sort(key=lambda x: x[0], reverse=True)
        return {
            "best": best, "best_prob": bp,
            "second": second, "second_prob": sp,
            "for": [e for e in ev if e[0] > 0.001],
            "against": [e for e in ev if e[0] < -0.001],
        }

    def confidence(self, obs: dict, method: str = "jsd") -> tuple:
        probs = self.proba(obs, method=method)
        sp = sorted(probs.values(), reverse=True)
        if not sp:
            return 0, "UNKNOWN"
        p0 = sp[0]
        gap = next((p0 - p for p in sp[1:] if p < p0 - 0.001), p0 - sp[1] if len(sp) > 1 else 0)
        label = "VERY HIGH" if gap > 0.5 else "HIGH" if gap > 0.3 else "MEDIUM" if gap > 0.15 else "LOW" if gap > 0.05 else "VERY LOW"
        return round(gap, 4), label

    def entropy_bits(self, obs: dict, method: str = "jsd") -> float:
        probs = self.proba(obs, method=method)
        h = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        return round(h, 3)
