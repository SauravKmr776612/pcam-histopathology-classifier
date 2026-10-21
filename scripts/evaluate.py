"""Evaluate a checkpoint on the held-out PCam test split."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcam_classifier.data import H5PcamDataset, find_h5_split, image_transform
from pcam_classifier.engine import run_epoch
from pcam_classifier.model import PcamCNN

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "pcam_cnn.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "test_metrics.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_file, y_file = find_h5_split(args.data_dir, "test")
    dataset = H5PcamDataset(x_file, y_file, image_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PcamCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
