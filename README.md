# LLM Fingerprint Classifier

Identify which language model produced a given set of answers — based on **single-token behavioral fingerprints**.

Built on the methodology and dataset from:

> **Bruckner, T.** (2026). *One Token Is Enough: Single-Token Output Distributions as Behavioral Fingerprints of Large Language Models.*  
> arXiv:2607.10252 — [https://arxiv.org/abs/2607.10252](https://arxiv.org/abs/2607.10252)  
> Dataset: [Zenodo DOI:10.5281/zenodo.21278557](https://doi.org/10.5281/zenodo.21278557)

## Usage

```bash
cd v3/
python3 main.py
```

Answer each question (comma-separated for multiple responses). Leave empty to skip.

```
Language [en/ru/zh/ar] (enter=en): en

Random number (1-100)   [en]: 42, 37, 57
Favorite number         [en]: 7
Coin flip               [en]: heads, heads, tails
Random animal           [en]: elephant
...
```

The classifier outputs:
- **Top-15 most likely models** with probabilities
- **Model family aggregation** (GPT, Claude, Qwen, …)
- **Evidence analysis** — which answers best distinguish the top model

Input normalization follows the paper: Unicode NFC, case folding, punctuation stripping, first-token extraction, color canonicalization, number-word mapping, and coin-flip normalization.

## Running the tests

```bash
python3 test_fingerprint.py
```

---

## Technical details

### Method

The classifier uses two complementary approaches from the paper:

1. **Jensen–Shannon Divergence (JSD)** — the paper's primary method (§IV-C). The empirical distribution of user-provided answers is compared against each model's known fingerprint distribution. Models are ranked by mean JSD across all answered cells.

2. **Naive Bayes** — a practical alternative that computes log-likelihood P(answers | model) with Laplace smoothing for unseen answers.

Both produce a probability distribution across all models via softmax normalization.

### Data

The fingerprint database contains 170 models fingerprinted through OpenRouter, comprising 10 tasks × 4 languages (English, Russian, Chinese, Arabic) with ≥30 samples per cell at temperature 1.0. Source data is from the paper's published Zenodo dataset.

### Tasks

| Task | Answer space |
|---|---|
| Random number 1–100 | closed (100) |
| Random number 1–10 | closed (10) |
| Favorite number | open |
| Random letter | closed (26) |
| Random word | open |
| Random color | open (canonicalized) |
| Favorite color | open (canonicalized) |
| Random animal | open |
| Random city | open |
| Coin flip | closed (2) |

### Normalization pipeline (§IV-B)

1. Unicode NFC normalization
2. Punctuation and quotation stripping
3. Case folding
4. First-token extraction
5. Task-specific canonicalization (digit mapping, color lexicon, coin-flip)

### Project structure

```
v3/
├── main.py               # Interactive CLI
├── fingerprint.py        # Normalization, classifier, JSD/Naive Bayes
├── test_fingerprint.py   # Test suite (6 suites)
├── data/
│   └── models.json       # 170 model fingerprints
└── README.md
```
