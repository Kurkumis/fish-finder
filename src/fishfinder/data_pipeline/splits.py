"""Train/validation/test splitting that keeps every observer inside one split."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

#test, validation, train: 14/14/71% split
N_FOLDS = 7


def assign_folds(
    df: pd.DataFrame,
    label_col: str = "class_id",
    group_col: str = "observer_id",
    n_folds: int = N_FOLDS,
    seed: int = 42,
) -> pd.Series:
    """Assign a fold number to every row.

    All rows with the same `group_col` value land in the same fold, while the
    class proportions are kept as even as possible across folds.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=df.index)
    for fold, (_, held_out) in enumerate(splitter.split(df, df[label_col], groups=df[group_col])):
        folds.iloc[held_out] = fold
    return folds


def fold_to_split(fold: int) -> str:
    """Map a fold number to a split name."""
    return {0: "test", 1: "val"}.get(fold, "train")
