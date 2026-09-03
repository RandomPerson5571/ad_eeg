"""Leakage-safe EEGNet training from exported epoch arrays.

The public entry point, :func:`run_deep_learning_benchmark`, is intentionally
not imported by ``eeg.training`` so classical-ML users do not need PyTorch.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from eeg.config import resolve_dataset
from eeg.contracts import validate_epoch_exports
from eeg.io import list_preprocessed_subjects, write_json
from eeg.paths import epochs_npy_dir, models_dir, results_dir
from eeg.repro import attach_repro_metadata, init_repro
from eeg.training.evaluation import (
    bootstrap_ci,
    compute_benchmark_metrics,
    plot_confusion_matrix,
    plot_pr,
    plot_roc,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise ImportError(
            "Deep-learning training requires PyTorch. On Kaggle, enable a GPU "
            "accelerator and use the preinstalled PyTorch runtime."
        ) from exc
    return torch


def _subject_table(dataset: str, experiment: str, paths: Iterable[Path]) -> pd.DataFrame:
    """Resolve one label for every exported subject and ignore absent subjects."""
    paths = [Path(path) for path in paths]
    available = {path.stem: path for path in paths}
    spec = resolve_dataset(dataset)[0]
    participants = list_preprocessed_subjects(spec, experiment).copy()
    if "participant_id" not in participants or "Group" not in participants:
        raise ValueError("Participant metadata needs participant_id and Group columns.")
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants = participants[participants["participant_id"].isin(available)].copy()
    participants = participants.drop_duplicates("participant_id")
    participants["path"] = participants["participant_id"].map(available)

    missing = sorted(set(available) - set(participants["participant_id"]))
    if missing:
        raise ValueError(f"No class label is available for exported subjects: {missing[:5]}")
    if len(participants) != len(available):
        raise ValueError("Participant metadata must contain exactly one row per epoch file.")
    return participants[["participant_id", "Group", "path"]].sort_values(
        "participant_id"
    ).reset_index(drop=True)


def _subject_folds(subjects: pd.DataFrame, n_splits: int, seed: int):
    class_counts = subjects["label_id"].value_counts()
    possible = int(class_counts.min())
    effective = min(int(n_splits), possible)
    if effective < 2:
        raise ValueError("Subject-grouped CV requires at least two subjects per class.")
    splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(subjects, subjects["label_id"]):
        yield subjects.iloc[train_idx].reset_index(drop=True), subjects.iloc[
            test_idx
        ].reset_index(drop=True)


def _train_validation_split(
    subjects: pd.DataFrame, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_idx, validation_idx = train_test_split(
            np.arange(len(subjects)),
            test_size=validation_size,
            random_state=seed,
            stratify=subjects["label_id"],
        )
    except ValueError as exc:
        raise ValueError(
            "The training fold is too small for a class-stratified subject-level "
            "validation split. Reduce validation_size or the number of CV folds."
        ) from exc
    return (
        subjects.iloc[train_idx].reset_index(drop=True),
        subjects.iloc[validation_idx].reset_index(drop=True),
    )


def _channel_statistics(subjects: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel normalization using training subjects only."""
    channel_sum = None
    channel_sq_sum = None
    count = 0
    for path in subjects["path"]:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, data.shape[0], 16):
            values = np.asarray(data[start : start + 16], dtype=np.float64)
            current_sum = values.sum(axis=(0, 2))
            current_sq_sum = np.einsum("ect,ect->c", values, values)
            channel_sum = (
                current_sum if channel_sum is None else channel_sum + current_sum
            )
            channel_sq_sum = (
                current_sq_sum
                if channel_sq_sum is None
                else channel_sq_sum + current_sq_sum
            )
            count += int(values.shape[0] * values.shape[2])
    if not count or channel_sum is None or channel_sq_sum is None:
        raise ValueError("Cannot normalize an empty training cohort.")
    mean = channel_sum / count
    variance = np.maximum(channel_sq_sum / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class EpochArrayDataset:
    """Lazy index over per-subject arrays; compatible with torch DataLoader."""

    def __init__(self, subjects: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.mean = mean[:, None]
        self.std = std[:, None]
        self.records: list[tuple[Path, int, int, str]] = []
        self.subject_epoch_counts: Counter[str] = Counter()
        self.subject_labels: dict[str, int] = {}
        for row in subjects.itertuples(index=False):
            array = np.load(row.path, mmap_mode="r", allow_pickle=False)
            self.subject_epoch_counts[row.participant_id] = int(array.shape[0])
            self.subject_labels[row.participant_id] = int(row.label_id)
            self.records.extend(
                (Path(row.path), epoch_idx, int(row.label_id), row.participant_id)
                for epoch_idx in range(array.shape[0])
            )
        # Memory maps do not load array contents eagerly. Keeping one per subject
        # avoids repeatedly reopening files when the balanced sampler interleaves
        # subjects.
        self._arrays: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        torch = _require_torch()
        path, epoch_idx, label, participant_id = self.records[index]
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        epoch = (np.asarray(self._arrays[path][epoch_idx]) - self.mean) / self.std
        # EEGNet consumes batch x 1 x channels x samples.
        return (
            torch.from_numpy(np.asarray(epoch, dtype=np.float32)[None, ...]),
            torch.tensor(label, dtype=torch.long),
            participant_id,
        )

    def balanced_sample_weights(self) -> np.ndarray:
        """Give every class equal mass, then every subject equal mass per class."""
        class_subjects = Counter(self.subject_labels.values())
        return np.asarray(
            [
                1.0
                / (
                    class_subjects[label]
                    * self.subject_epoch_counts[participant_id]
                )
                for _, _, label, participant_id in self.records
            ],
            dtype=np.float64,
        )


def build_eegnet(
    n_channels: int,
    n_classes: int,
    *,
    temporal_kernel: int = 63,
    dropout: float = 0.5,
):
    """Build a compact EEGNet-style network for (B, 1, C, T) input."""
    torch = _require_torch()
    nn = torch.nn
    f1, depth_multiplier, f2 = 8, 2, 16

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(
                    1,
                    f1,
                    kernel_size=(1, temporal_kernel),
                    padding=(0, temporal_kernel // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(f1),
                nn.Conv2d(
                    f1,
                    f1 * depth_multiplier,
                    kernel_size=(n_channels, 1),
                    groups=f1,
                    bias=False,
                ),
                nn.BatchNorm2d(f1 * depth_multiplier),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(
                    f1 * depth_multiplier,
                    f1 * depth_multiplier,
                    kernel_size=(1, 15),
                    padding=(0, 7),
                    groups=f1 * depth_multiplier,
                    bias=False,
                ),
                nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=1, bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(f2, n_classes)

        def forward(self, inputs):
            return self.classifier(self.features(inputs).flatten(1))

    return EEGNet()


def _loader(dataset, batch_size: int, *, training: bool, seed: int):
    torch = _require_torch()
    if training:
        generator = torch.Generator().manual_seed(seed)
        sampler = torch.utils.data.WeightedRandomSampler(
            dataset.balanced_sample_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _subject_predictions(model, loader, device, n_classes: int) -> pd.DataFrame:
    torch = _require_torch()
    probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for inputs, targets, participant_ids in loader:
            batch_probability = torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
            for participant_id, target, probability in zip(
                participant_ids, targets.numpy(), batch_probability
            ):
                labels[participant_id] = int(target)
                probabilities[participant_id].append(probability)
    rows = []
    for participant_id in sorted(probabilities):
        mean_probability = np.mean(probabilities[participant_id], axis=0)
        row = {
            "participant_id": participant_id,
            "y_true": labels[participant_id],
            "y_pred": int(mean_probability.argmax()),
            "n_epochs": len(probabilities[participant_id]),
        }
        row.update({f"proba_{idx}": float(mean_probability[idx]) for idx in range(n_classes)})
        rows.append(row)
    return pd.DataFrame(rows)


def _balanced_accuracy(frame: pd.DataFrame) -> float:
    return compute_benchmark_metrics(frame["y_true"], frame["y_pred"])[
        "balanced_accuracy"
    ]


def _fit_fold(
    train_subjects: pd.DataFrame,
    validation_subjects: pd.DataFrame,
    *,
    n_channels: int,
    n_classes: int,
    device,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
):
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean, std = _channel_statistics(train_subjects)
    train_data = EpochArrayDataset(train_subjects, mean, std)
    validation_data = EpochArrayDataset(validation_subjects, mean, std)
    train_loader = _loader(train_data, batch_size, training=True, seed=seed)
    validation_loader = _loader(validation_data, batch_size, training=False, seed=seed)
    model = build_eegnet(n_channels, n_classes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_score = -np.inf
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_total = 0.0
        sample_count = 0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(targets)
            sample_count += len(targets)

        validation_predictions = _subject_predictions(
            model, validation_loader, device, n_classes
        )
        validation_score = _balanced_accuracy(validation_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / max(sample_count, 1),
                "validation_balanced_accuracy": validation_score,
            }
        )
        if validation_score > best_score + 1e-6:
            best_score = validation_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    return model, mean, std, history, float(best_score)


def run_deep_learning_benchmark(
    dataset: str,
    experiment: str,
    *,
    cv_folds: int = 5,
    validation_size: float = 0.2,
    max_epochs: int = 30,
    patience: int = 6,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train EEGNet with subject OOF evaluation and save compact artifacts."""
    torch = _require_torch()
    init_repro(seed)
    epoch_paths = sorted(epochs_npy_dir(dataset, experiment).glob("sub-*.npy"))
    contract = validate_epoch_exports(epoch_paths)
    subjects = _subject_table(dataset, experiment, epoch_paths)
    encoder = LabelEncoder()
    subjects["label_id"] = encoder.fit_transform(subjects["Group"])
    class_names = list(encoder.classes_)
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("Deep-learning classification requires at least two classes.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch_device = torch.device(resolved_device)
    fold_predictions = []
    fold_details = []
    start = time.perf_counter()

    folds = list(_subject_folds(subjects, cv_folds, seed))
    for fold_idx, (outer_train, outer_test) in enumerate(folds):
        fold_seed = seed + fold_idx
        train_subjects, validation_subjects = _train_validation_split(
            outer_train, validation_size, fold_seed
        )
        model, mean, std, history, best_validation_score = _fit_fold(
            train_subjects,
            validation_subjects,
            n_channels=contract["n_channels"],
            n_classes=n_classes,
            device=torch_device,
            seed=fold_seed,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        test_data = EpochArrayDataset(outer_test, mean, std)
        test_loader = _loader(test_data, batch_size, training=False, seed=fold_seed)
        predictions = _subject_predictions(model, test_loader, torch_device, n_classes)
        predictions.insert(0, "fold", fold_idx)
        predictions.insert(0, "model", "eegnet")
        fold_predictions.append(predictions)
        fold_details.append(
            {
                "fold": fold_idx,
                "train_subjects": sorted(train_subjects["participant_id"]),
                "validation_subjects": sorted(validation_subjects["participant_id"]),
                "test_subjects": sorted(outer_test["participant_id"]),
                "best_validation_balanced_accuracy": best_validation_score,
                "epochs_trained": len(history),
                "history": history,
            }
        )
        print(
            f"Fold {fold_idx + 1}/{len(folds)}: "
            f"validation={best_validation_score:.3f}, "
            f"test={_balanced_accuracy(predictions):.3f}, epochs={len(history)}",
            flush=True,
        )

    oof = pd.concat(fold_predictions, ignore_index=True).sort_values("participant_id")
    if oof["participant_id"].duplicated().any() or set(oof["participant_id"]) != set(
        subjects["participant_id"]
    ):
        raise RuntimeError("Outer CV must produce exactly one prediction per subject.")
    probability_columns = [f"proba_{idx}" for idx in range(n_classes)]
    probabilities = oof[probability_columns].to_numpy()
    metrics = compute_benchmark_metrics(
        oof["y_true"].to_numpy(),
        oof["y_pred"].to_numpy(),
        probabilities,
        labels=list(range(n_classes)),
    )

    def metric_fn(y_true, y_pred, y_probability):
        return compute_benchmark_metrics(y_true, y_pred)["balanced_accuracy"]

    _, ci_low, ci_high = bootstrap_ci(
        metric_fn,
        oof["y_true"].to_numpy(),
        oof["y_pred"].to_numpy(),
        probabilities,
        n=bootstrap_iterations,
        seed=seed,
        strata=oof["y_true"].to_numpy(),
    )
    runtime = time.perf_counter() - start

    result_root = results_dir(dataset, experiment)
    model_root = models_dir(dataset, experiment)
    figure_root = result_root / "figures"
    result_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    oof["y_true_label"] = encoder.inverse_transform(oof["y_true"].to_numpy())
    oof["y_pred_label"] = encoder.inverse_transform(oof["y_pred"].to_numpy())
    predictions_path = result_root / "deep_learning_predictions.csv"
    oof.to_csv(predictions_path, index=False)

    benchmark_row = {
        "model": "eegnet",
        "dataset": dataset,
        "experiment": experiment,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_roc_auc": metrics.get("macro_roc_auc"),
        "macro_pr_auc": metrics.get("macro_pr_auc"),
        "mcc": metrics["mcc"],
        "cohen_kappa": metrics["cohen_kappa"],
        "balanced_accuracy_ci_lo": ci_low,
        "balanced_accuracy_ci_hi": ci_high,
        "runtime_s": runtime,
        "n_folds": len(folds),
        "n_oof_subjects": len(oof),
        "device": str(torch_device),
    }
    benchmark_path = result_root / "deep_learning_benchmark.csv"
    pd.DataFrame([benchmark_row]).to_csv(benchmark_path, index=False)
    detail_path = result_root / "deep_learning_detail.json"
    write_json(
        detail_path,
        {
            "class_names": class_names,
            "aggregate_subject_oof": metrics,
            "fold_details": fold_details,
        },
    )
    metadata = attach_repro_metadata(
        {
            "dataset": dataset,
            "experiment_name": experiment,
            "model": "eegnet",
            "evaluation": "stratified_subject_cv_subject_oof",
            "input_contract": contract,
        },
        {
            "experiment_name": experiment,
            "seed": seed,
            "deep_learning": {
                "cv_folds": cv_folds,
                "validation_size": validation_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "dropout": dropout,
            },
        },
    )
    write_json(result_root / "deep_learning_metadata.json", metadata)
    plot_confusion_matrix(
        oof["y_true"], oof["y_pred"], class_names, figure_root / "confusion_eegnet.png"
    )
    plot_roc(oof["y_true"], probabilities, class_names, figure_root / "roc_eegnet.png")
    plot_pr(oof["y_true"], probabilities, class_names, figure_root / "pr_eegnet.png")

    # Save a deployment model fitted with a subject-level validation cohort.
    final_train, final_validation = _train_validation_split(subjects, validation_size, seed)
    final_model, final_mean, final_std, final_history, final_score = _fit_fold(
        final_train,
        final_validation,
        n_channels=contract["n_channels"],
        n_classes=n_classes,
        device=torch_device,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
    )
    model_path = model_root / "eegnet.pt"
    torch.save(
        {
            "state_dict": final_model.cpu().state_dict(),
            "n_channels": contract["n_channels"],
            "n_samples": contract["n_samples"],
            "class_names": class_names,
            "channel_mean": final_mean,
            "channel_std": final_std,
            "validation_balanced_accuracy": final_score,
            "epochs_trained": len(final_history),
        },
        model_path,
    )
    return {
        "benchmark_csv": str(benchmark_path),
        "predictions_csv": str(predictions_path),
        "detail_json": str(detail_path),
        "model_path": str(model_path),
        "row": benchmark_row,
    }
