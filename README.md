# PCam Histopathology Patch Classifier

A small, end-to-end computer vision portfolio project for binary classification of histopathology image patches using PyTorch. It includes HDF5 dataset loading, image preprocessing and augmentation, CNN training, held-out evaluation, and a Streamlit demo.

**This is an educational project, not a medical device. Predictions must not be used for diagnosis or patient care.** No dataset images or trained weights are included in this repository.

## What it does

- Loads official PCam train, validation, and test splits from matching HDF5 files.
- Resizes patches to 96 × 96, normalizes RGB values, and applies random flips during training.
- Trains a compact CNN and selects a checkpoint using validation ROC AUC.
- Reports loss, accuracy, precision, recall, F1, ROC AUC, and a confusion matrix. Results are generated locally; this repository makes no performance claims.
- Offers a one-image Streamlit demo for a locally trained checkpoint.

## Requirements

Python 3.10 or newer is recommended. Training the full dataset can take substantial disk space and time; a CUDA-capable GPU is optional. Start with `--limit 2000` to exercise the pipeline on a subset.

### Install (Windows PowerShell)

```powershell
cd path\to\pcam-histopathology-classifier
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For other operating systems, create and activate a Python virtual environment with `python -m venv .venv`, then install `requirements.txt`.

If PyTorch cannot find a compatible accelerator, use the install selector on the [official PyTorch site](https://pytorch.org/get-started/locally/) to choose the right build for your machine.

## Dataset download and format

PCam is a separate dataset with its own terms and attribution. Download it yourself from the [upstream PCam project](https://github.com/basveeling/pcam) or its [Zenodo record](https://zenodo.org/records/2546921). Review the dataset page and applicable terms before use. This repository does not redistribute data.

Download the six gzipped HDF5 files for train, valid, and test (each split has an image file ending `_x.h5.gz` and labels ending `_y.h5.gz`). Extract the `.gz` files and put the resulting six `.h5` files directly in `data/raw/`, keeping their filenames. The loader expects:

| File suffix | HDF5 key | Contents |
| --- | --- | --- |
| `*_x.h5` | `x` | RGB images, normally 96 × 96 × 3 |
| `*_y.h5` | `y` | Binary labels, 0 or 1 |

Example names include `camelyonpatch_level_2_split_train_x.h5`, `camelyonpatch_level_2_split_train_y.h5`, and corresponding `valid` and `test` files. The validation filename uses `valid` (not `val`). Do not commit data files; `data/raw/` is ignored by Git.

## Train and evaluate

From the project root, after placing the files in `data/raw/`:

```powershell
python scripts/train.py --epochs 5 --batch-size 64 --limit 2000
python scripts/evaluate.py
```

Omit `--limit 2000` to use each complete split. Training saves the best validation checkpoint to `models/pcam_cnn.pt` and epoch metrics to `reports/training_history.json`. Test evaluation saves `reports/test_metrics.json`. These artifacts are ignored by Git so they stay local.

Options include `--epochs`, `--batch-size`, `--workers`, `--lr`, and `--data-dir`. Run `python scripts/train.py --help` or `python scripts/evaluate.py --help` for details.

## Run the demo

After training:

```powershell
streamlit run app.py
```

Upload one image patch. The demo resizes it to the model input size and displays a predicted class probability. Performance on arbitrary images may differ from PCam; the output is not clinically validated.

## Project layout

```text
pcam-histopathology-classifier/
├── app.py
├── data/
│   ├── raw/                 # Downloaded HDF5 data (ignored by Git)
│   └── processed/           # Reserved for derived data (ignored by Git)
├── models/                  # Local checkpoints (ignored by Git)
├── reports/                 # Local metrics (ignored by Git)
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/pcam_classifier/
│   ├── data.py              # HDF5 dataset and preprocessing
│   ├── engine.py            # Training/evaluation metrics
│   └── model.py             # CNN architecture
├── requirements.txt
└── .gitignore
```



## Limitations

- PCam labels describe patches, not whole slides or patients. The benchmark task focuses on tumor presence in the central region of a patch.
- A simple CNN and one benchmark split do not establish generalization to other scanners, stains, institutions, or clinical populations.
- The demo does not provide calibration, uncertainty estimates, clinical workflow integration, or diagnostic guidance.
- Dataset availability, terms, and hosting can change; consult the upstream source before downloading or using it.
