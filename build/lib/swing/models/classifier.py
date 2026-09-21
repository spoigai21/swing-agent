"""Phase 5.2 — the transformer that must beat the TF-IDF baseline, or be dropped.

agent-plan.md Step 5.2: "Fine-tune a small encoder with `transformers`
(DistilBERT-scale is plenty)... Baseline to beat: LightGBM on TF-IDF. If your
transformer cannot beat it, you do not have enough labels yet — go label more
instead of tuning hyperparameters."

    .venv/bin/python -m swing.models.classifier

Gate 5 is pass-or-drop: "Each of the three components beats its baseline on a
held-out, time-forward split. Any that doesn't, drop — the heuristic version is
fine." So this module exists to produce a number that can say "no". It trains on
the SAME split as `models.baseline`, from the SAME leakage-stripped text, so the
comparison is like for like.

⚠️ 608 training rows across 4 classes is small for a 66M-parameter encoder, and
TF-IDF is a genuinely strong baseline on formulaic text like filings. A loss
here is the expected outcome and is informative: it means label more, not tune
more. Nothing ships on a tie.
"""
from __future__ import annotations

from dataclasses import dataclass

from swing.models.baseline import MIN_WORDS, TRAIN_FRAC
from swing.models.events import dataset, distribution
from swing.models.splits import split_at

MODEL = "distilbert-base-uncased"
MAX_LEN = 128
BATCH = 16
EPOCHS = 4
LR = 3e-5
SEED = 0          # reproducibility of INITIALISATION only; the split is never shuffled


@dataclass(frozen=True, slots=True)
class Result:
    accuracy: float
    macro_f1: float
    epochs: int
    train_rows: int
    test_rows: int
    per_class: dict[str, float] | None = None
    support: dict[str, int] | None = None

    def by_class(self) -> str:
        """⚠️ macro-F1 weights a 13-row class the same as a 500-row one, so a
        few rows flipping in `m_and_a` can move it more than the entire margin
        over the baseline. Always read this alongside the headline number."""
        if not self.per_class:
            return ""
        sup = self.support or {}
        return "    " + "  ".join(
            f"{k}={v:.2f}(n={sup.get(k, 0)})" for k, v in sorted(self.per_class.items()))


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(min_words: int = MIN_WORDS, train_frac: float = TRAIN_FRAC,
        epochs: int = EPOCHS, seed: int = SEED) -> Result:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)

    rows = dataset(min_words=min_words)
    dates = sorted(r.published_at for r in rows)
    split = split_at(rows, dates[int(len(dates) * train_frac)])
    labels = sorted(distribution(rows))
    index = {name: i for i, name in enumerate(labels)}

    device = _device()
    print(f"{len(rows)} rows, {len(labels)} classes, device={device}")
    print(f"  {split} (cutoff {split.cutoff:%Y-%m-%d}, time-forward)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels)).to(device)

    def encode(part):
        enc = tok([r.text for r in part], truncation=True, padding="max_length",
                  max_length=MAX_LEN, return_tensors="pt")
        y = torch.tensor([index[r.label] for r in part])
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    # shuffle=True here reorders rows WITHIN the training set only; it does not
    # move anything across the time boundary, which models.splits enforces.
    train_dl = DataLoader(encode(split.train), batch_size=BATCH, shuffle=True)
    test_dl = DataLoader(encode(split.test), batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for ids, mask, y in train_dl:
            opt.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=y.to(device))
            out.loss.backward()
            opt.step()
            total += out.loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  loss={total / max(1, len(train_dl)):.4f}",
              flush=True)

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for ids, mask, y in test_dl:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            golds.extend(y.tolist())

    from collections import Counter

    order = list(range(len(labels)))
    each = f1_score(golds, preds, average=None, labels=order, zero_division=0)
    counts = Counter(golds)
    return Result(accuracy=accuracy_score(golds, preds),
                  macro_f1=f1_score(golds, preds, average="macro", zero_division=0),
                  epochs=epochs, train_rows=len(split.train), test_rows=len(split.test),
                  per_class={labels[i]: float(each[i]) for i in order},
                  support={labels[i]: counts.get(i, 0) for i in order})


SEEDS = (0, 1, 2)


def one_row_resolution(support: dict[str, int]) -> float:
    """How much macro-F1 moves when ONE test row of the smallest class flips.

    macro-F1 weights every class equally, so with C classes and a smallest class
    of n test rows a single row is worth (1/C)/n. Here that is (1/5)/14 = 0.0143
    for `m_and_a` — larger than the 0.010 margin by which distilbert first
    appeared to "beat the baseline on every seed". A margin below this is finer
    than the instrument and cannot be called a result.
    """
    if not support:
        return 0.0
    return (1 / len(support)) / max(1, min(support.values()))


def verdict(margin: float, resolution: float, mean_got: float, mean_base: float) -> str:
    """One of: pass | below_resolution | inside_noise | loses.

    ⚠️ This rule has been tightened twice, both times because it announced a
    pass it had not earned — first on a single seed against a single baseline
    run, then on a margin smaller than one test row. Both readings would have
    gone into the record as "Gate 5 satisfied".
    """
    if margin > resolution:
        return "pass"
    if margin > 0:
        return "below_resolution"
    return "inside_noise" if mean_got > mean_base else "loses"


VERDICT_TEXT = {
    "pass": "beats the baseline by more than one row of the smallest class, on every "
            "seed. Gate 5 satisfied for 5.2.",
    "below_resolution": "ahead on every seed but by LESS than a single test row of the "
                        "smallest class — below the resolution of this test set. Not a "
                        "pass: agent-plan.md 5.2 reads that as too few labels. Keep "
                        "TF-IDF, label more.",
    "inside_noise": "ahead on average but INSIDE the noise. Not a pass — agent-plan.md "
                    "5.2 says that means label more, not tune more. Keep TF-IDF.",
    "loses": "does not beat the baseline. Keep TF-IDF, go label more.",
}


def main() -> int:
    """Compare DISTRIBUTIONS, not single runs.

    ⚠️ The first version of this declared "Gate 5 satisfied" on any margin at
    all. It reported 0.801 vs 0.787 — but the boosting baseline moves on its own
    between identical runs (SVD and the trees carry no fixed seed), and the test
    set is 153 rows across 4 classes. A margin of 0.014 under those conditions is
    noise wearing a result's clothes. So: several runs of each, and the
    transformer has to clear the baseline's best run, not its average.
    """
    import statistics as st
    from contextlib import redirect_stdout
    from io import StringIO

    from swing.models.baseline import run as baseline_run

    base: list[float] = []
    for _ in range(5):
        with redirect_stdout(StringIO()):
            base.append(max(baseline_run()[1:], key=lambda s: s.macro_f1).macro_f1)
    print(f"  baseline (5 runs)    macroF1 mean={st.mean(base):.3f} "
          f"sd={st.pstdev(base):.3f} max={max(base):.3f}")

    results = [run(seed=s) for s in SEEDS]
    for s, r in zip(SEEDS, results, strict=True):
        print(f"  seed {s}: macroF1={r.macro_f1:.3f}")
        print(r.by_class())
    got = [r.macro_f1 for r in results]
    print(f"  distilbert ({len(SEEDS)} seeds) macroF1 mean={st.mean(got):.3f} "
          f"sd={st.pstdev(got):.3f} min={min(got):.3f} max={max(got):.3f}")

    # ⚠️ A margin is meaningless if one test row outweighs it. macro-F1 weights
    # every class equally, so with C classes and a smallest class of n test rows,
    # flipping ONE row moves the score by (1/C)/n. Here that is (1/5)/14 =
    # 0.0143 for m_and_a, while distilbert cleared the baseline by 0.010 — the
    # "win" is finer than the instrument. Clearing the best baseline run is
    # necessary but not sufficient; the margin has to exceed one row too.
    support = results[0].support or {}
    resolution = one_row_resolution(support)
    margin = min(got) - max(base)
    smallest = min(support, key=support.get) if support else "?"
    print(f"\n  margin at worst seed = {margin:+.3f};  one row of the smallest class "
          f"('{smallest}', n={support.get(smallest, 0)}) is worth {resolution:.3f}")
    print("  VERDICT: " + VERDICT_TEXT[
        verdict(margin, resolution, st.mean(got), st.mean(base))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
