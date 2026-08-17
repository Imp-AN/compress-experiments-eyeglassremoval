import copy
import time
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from compression import apply_pruning, finetune


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)

    l1_total = 0.0
    mse_total = 0.0
    n = 0

    for x, target in loader:
        x = x.to(device)
        target = target.to(device)
        pred = model(x)

        l1_total += F.l1_loss(pred, target, reduction="sum").item()
        mse_total += F.mse_loss(pred, target, reduction="sum").item()
        n += target.numel()

    l1 = l1_total / n
    mse = mse_total / n
    psnr = 10 * torch.log10(torch.tensor(4.0 / mse)).item()  # data range is 2 for [-1, 1]

    return {"L1": l1, "PSNR": psnr}


@torch.no_grad()
def benchmark(model, x, device, runs=50):
    model.eval().to(device)
    x = x.to(device)

    for _ in range(10):
        model(x)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        model(x)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / runs


def cross_validate_pruning(
    base_model,
    tuning_dataset,
    configs,
    device,
    n_splits=3,
    batch_size=8,
    finetune_epochs=1,
    lr=1e-5,
    random_state=42,
):
    """Choose pruning hyperparameters using CV on tuning data only; keep test data separate."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for config in configs:
        fold_scores = []

        for train_idx, val_idx in kf.split(tuning_dataset):
            train_loader = DataLoader(
                Subset(tuning_dataset, train_idx),
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                Subset(tuning_dataset, val_idx),
                batch_size=batch_size,
                shuffle=False,
            )

            # Fresh copy of the same pretrained baseline for every fold.
            model = apply_pruning(copy.deepcopy(base_model), config)
            model = finetune(model, train_loader, device, epochs=finetune_epochs, lr=lr)
            score = evaluate(model, val_loader, device)
            fold_scores.append(score)

        results.append({
            **config,
            "L1": sum(s["L1"] for s in fold_scores) / len(fold_scores),
            "PSNR": sum(s["PSNR"] for s in fold_scores) / len(fold_scores),
            "folds": fold_scores,
        })

    return results
