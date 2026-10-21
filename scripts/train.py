"""Train the PCam CNN. See README.md for data download and commands."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcam_classifier.data import H5PcamDataset, find_h5_split, image_transform
from pcam_classifier.engine import run_epoch
from pcam_classifier.model import PcamCNN

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum examples per split")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "pcam_cnn.pt")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx, ty = find_h5_split(args.data_dir, "train")
    vx, vy = find_h5_split(args.data_dir, "valid")
    train_ds = H5PcamDataset(tx, ty, image_transform(train=True))
    val_ds = H5PcamDataset(vx, vy, image_transform())
    if args.limit > 0:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    model = PcamCNN().to(device)
    optimizer, criterion = torch.optim.AdamW(model.parameters(), lr=args.lr), nn.CrossEntropyLoss()
    best_auc, history = float("-inf"), []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(json.dumps(record))
        if val_metrics["roc_auc"] >= best_auc:
            best_auc = val_metrics["roc_auc"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_metrics": val_metrics, "image_size": 96}, args.output)
    report = ROOT / "reports" / "training_history.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Checkpoint: {args.output}\nHistory: {report}\nDevice: {device}")

if __name__ == "__main__":
    main()
