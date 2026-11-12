"""Evaluate frozen embeddings with grouped cross-validation (kNN and logistic regression)."""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishfinder.utils.config import load_backbones, load_classes
from fishfinder.utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR

DEV_FOLDS = [1, 2, 3, 4, 5, 6]  # fold 0 is the test set and stays untouched


def knn_predict(train_x, train_y, test_x, n_classes, k=10, temperature=0.07):
    """Weighted k-nearest-neighbour vote on L2-normalized features."""
    sims = test_x @ train_x.T  # cosine similarities, shape (n_test, n_train)
    nearest = np.argpartition(-sims, k, axis=1)[:, :k]  # indices of the k closest photos
    nearest_sims = np.take_along_axis(sims, nearest, axis=1)
    weights = np.exp(nearest_sims / temperature)  # closer neighbours vote more strongly
    votes = np.zeros((len(test_x), n_classes))
    for c in range(n_classes):
        votes[:, c] = (weights * (train_y[nearest] == c)).sum(axis=1)
    return votes.argmax(axis=1)


def top3_accuracy(proba: np.ndarray, y: np.ndarray) -> float:
    top3 = np.argsort(-proba, axis=1)[:, :3]
    return float((top3 == y[:, None]).any(axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-mode", default="squash")
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    n_classes = len(load_classes())
    dev_mask = df["fold"].isin(DEV_FOLDS).to_numpy()
    y_all = df["class_index"].to_numpy()
    folds_all = df["fold"].to_numpy()

    emb_dir = ARTIFACTS_DIR / "embeddings" / args.crop_mode
    oof_dir = ARTIFACTS_DIR / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    summary, per_class = [], {}
    for entry in load_backbones():
        name = entry["name"]
        path = emb_dir / f"{name}.npy"
        if not path.exists():
            print(f"skip {name}: no embeddings")
            continue

        emb = np.load(path)
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        oof_proba = np.zeros((len(df), n_classes))
        oof_knn = np.full(len(df), -1)
        fold_scores = []

        for fold in DEV_FOLDS:
            test = (folds_all == fold)
            train = dev_mask & ~test
            # kNN on normalized features
            pred_knn = knn_predict(emb_norm[train], y_all[train], emb_norm[test], n_classes)
            oof_knn[test] = pred_knn
            # logistic regression on standardized features
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            clf.fit(emb[train], y_all[train])
            proba = clf.predict_proba(emb[test])
            oof_proba[test] = proba
            pred_lr = proba.argmax(axis=1)
            fold_scores.append(
                {
                    "knn_acc": accuracy_score(y_all[test], pred_knn),
                    "lr_acc": accuracy_score(y_all[test], pred_lr),
                    "lr_top3": top3_accuracy(proba, y_all[test]),
                    "lr_macro_recall": balanced_accuracy_score(y_all[test], pred_lr),
                }
            )

        scores = pd.DataFrame(fold_scores)
        row = {"model": name, "dim": emb.shape[1]}
        for col in scores:
            row[col] = f"{scores[col].mean():.3f} ± {scores[col].std():.3f}"
        summary.append(row)

        # Out-of-fold predictions are saved for the label-issue analysis later.
        np.savez(oof_dir / f"{name}.npz", proba=oof_proba, knn_pred=oof_knn, dev_mask=dev_mask)
        # Per-class recall of logistic regression (out-of-fold, development folds only).
        lr_pred = oof_proba.argmax(axis=1)
        per_class[name] = pd.Series(
            {c: (lr_pred[dev_mask & (y_all == c)] == c).mean() for c in range(n_classes)}
        )
        print(f"done: {name}")

    table = pd.DataFrame(summary)
    print("\n" + table.to_string(index=False))

    REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORTS_DIR / "frozen_screening.csv", index=False)
    class_ids = [c["id"] for c in load_classes()]
    per_class_df = pd.DataFrame(per_class)
    per_class_df.index = class_ids
    per_class_df.round(3).to_csv(REPORTS_DIR / "frozen_screening_per_class.csv")
    print("\nPer-class recall (logistic regression):")
    print(per_class_df.round(2).to_string())


if __name__ == "__main__":
    main()
