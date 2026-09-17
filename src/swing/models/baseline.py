"""Phase 5.2 baseline: TF-IDF, run BEFORE any transformer.

agent-plan.md Step 5.2: "Baseline to beat: LightGBM on TF-IDF. **Run the
baseline first.** It is 10 minutes of work and frequently wins on small
datasets. If your transformer cannot beat it, you do not have enough labels yet
— go label more instead of tuning hyperparameters."

    .venv/bin/python -m swing.models.baseline

⚠️ Substitution: `lightgbm` is installed but cannot load on this machine — it
needs `libomp.dylib` (Homebrew's `libomp`), which is absent. sklearn's
`HistGradientBoostingClassifier` is the same family of gradient-boosted trees
with no OpenMP dependency, so it stands in. If libomp is ever installed, swap it
back; the numbers should be close either way.

⚠️ Two things this measures honestly, both learned the hard way:

1. **The split is time-forward** (`models.splits`), never shuffled. Filings are
   formulaic and repetitive across years, so a shuffled split would put a
   company's Q2 filing in training and its Q3 in test and score beautifully.
2. **Item numbers are stripped from the input** (`events.strip_label_leakage`).
   The labels are derived from those numbers and EDGAR prints them in the
   headline, so leaving them in scored 0.900 accuracy for reading the answer off
   the input, against 0.835 without.
"""
from __future__ import annotations

from dataclasses import dataclass

from swing.models.events import dataset, distribution
from swing.models.splits import split_at

MIN_WORDS = 5          # below this a filing is pure boilerplate: a label, no evidence
SEED = 0               # fixes the ESTIMATORS, never the split (see models.splits)
TRAIN_FRAC = 0.8


@dataclass(frozen=True, slots=True)
class Score:
    name: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, float] | None = None

    def line(self) -> str:
        return f"  {self.name:<32} acc={self.accuracy:.3f}  macroF1={self.macro_f1:.3f}"

    def by_class(self) -> str:
        """⚠️ Read this, not the macro number, once analyst_action is included:
        it is 77% of the rows and trivially separable (every row is templated),
        so it lifts macro-F1 for every model equally and hides what happens to
        the small, hard classes."""
        if not self.per_class:
            return ""
        return "    " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(self.per_class.items()))


def run(min_words: int = MIN_WORDS, train_frac: float = TRAIN_FRAC) -> list[Score]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    rows = dataset(min_words=min_words)
    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} labelled rows with >= {min_words} words; "
                         "not enough to measure anything")
    dates = sorted(r.published_at for r in rows)
    split = split_at(rows, dates[int(len(dates) * train_frac)])
    y_train = [r.label for r in split.train]
    y_test = [r.label for r in split.test]

    print(f"{len(rows)} labelled events with >= {min_words} real words")
    print(f"  {split}  (cutoff {split.cutoff:%Y-%m-%d}, time-forward)")
    print(f"  labels: {distribution(rows)}")

    classes = sorted(set(y_train) | set(y_test))

    def scored(name: str, pred) -> Score:
        each = f1_score(y_test, pred, average=None, labels=classes, zero_division=0)
        return Score(name, accuracy_score(y_test, pred),
                     f1_score(y_test, pred, average="macro", zero_division=0),
                     per_class=dict(zip(classes, (float(v) for v in each), strict=True)))

    # The floor is the commonest class IN TRAINING — using the overall majority
    # would peek at the test period.
    major = max(distribution(split.train).items(), key=lambda kv: kv[1])[0]
    out = [scored(f"majority class ('{major}')", [major] * len(y_test))]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          strip_accents="unicode")
    x_train = vec.fit_transform([r.text for r in split.train])
    x_test = vec.transform([r.text for r in split.test])

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_train, y_train)
    out.append(scored("tf-idf + logistic regression", lr.predict(x_test)))

    # ⚠️ random_state is REQUIRED here, and it is not the thing models.splits
    # forbids: that bans shuffling the TIME ORDER, this makes a fitted model
    # reproducible. Without it this baseline scored 0.772-0.829 macroF1 across
    # seven identical runs (sd 0.020), a 0.057 spread -- four times the margin a
    # transformer was about to be credited with beating it by. A measuring stick
    # that moves cannot referee Gate 5.
    svd = TruncatedSVD(n_components=min(200, x_train.shape[1] - 1),
                       random_state=SEED).fit(x_train)
    gbm = HistGradientBoostingClassifier(max_iter=300,
                                         random_state=SEED).fit(svd.transform(x_train), y_train)
    out.append(scored("tf-idf + SVD + grad boosting", gbm.predict(svd.transform(x_test))))
    return out


def main() -> int:
    scores = run()
    print()
    for s in scores:
        print(s.line())
        if s.by_class():
            print(s.by_class())
    best = max(scores[1:], key=lambda s: s.macro_f1)
    print(f"\nA transformer must beat macroF1={best.macro_f1:.3f} ({best.name}) to ship. "
          "agent-plan.md Gate 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
