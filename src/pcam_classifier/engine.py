"""Shared train and evaluation loop."""
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, labels, preds, scores = 0.0, [], [], []
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            labels.extend(targets.cpu().tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / max(1, len(labels)), "accuracy": accuracy_score(labels, preds),
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc,
            "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist()}
