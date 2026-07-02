# PRICoD

## Repository structure

```
PRICoD/
├── train.py                 # training + inference entry point
├── calculate_metrics.py     # DSC / ASSD / HD / volume evaluation
├── SegModel.py              # ProjectUNet: ResNet-50 encoder + U-Net decoder + projector
├── requirements.txt
├── modules/                 # the three core contributions
│   ├── pri.py               #   PRI  (pixel-region-instance contrast)
│   ├── crm.py               #   CRM  (class-conditioned logit refinement)
│   └── urc.py               #   URC  (uncertainty-rectified consistency loss)
├── base/                    # segmentation-model scaffolding (heads, blocks, init)
├── encoders/                # ResNet-50 encoder
├── decoders/                # U-Net decoder
├── utilities/               # dataset, augmentation, evaluation helpers
│   ├── MyDataSet.py         #   SemiSegDataset (labeled/unlabeled split)
│   ├── Load_Data_v2.py      #   online 2D affine augmentation
│   ├── val_2D.py            #   volume-wise validation metrics
│   ├── losses.py            #   Dice loss
│   └── image_utils.py       #   DSC / ASSD / HD / volume metrics
├── data_dir/                # train/valid/test subject lists (CSV)
├── data/                    # <- place the raw imaging data here (see below)
└── scripts/
    ├── train_ACDC_10.sh     # ACDC,    10% labels
    └── train_MMs2_10.sh     # M&Ms-2,  10% labels
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is required. Tested with Python 3.10 and PyTorch >= 1.12.

## Data preparation

Volumes are read from `./data`. Each subject is a folder containing a
resampled image and its ground-truth label (`*_gt.nii.gz`); the subject lists in
`data_dir/` reference these files by relative path. Arrange the data as:

```
data/
├── ACDC/
│   ├── training/            # subjects used by train_ACDC.csv + valid_ACDC.csv
│   │   ├── patient001/patient001_ED_resamp.nii.gz
│   │   ├── patient001/patient001_ED_resamp_gt.nii.gz
│   │   └── ...
│   └── testing/             # subjects listed in test_subj_ACDC.csv
│       └── patient101/...
└── MnM-2/
    └── all/                 # all M&Ms-2 subjects (train/valid/test split by CSV)
        ├── 001/001_LA_ED_resamp.nii.gz
        ├── 001/001_LA_ED_resamp_gt.nii.gz
        └── ...
```

The public datasets are the [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
and [M&amp;Ms-2](https://www.ub.edu/mnms-2/) cardiac MRI challenges.

## Training and inference

Run one of the provided scripts (each trains the full model, then writes
predictions):

```bash
bash scripts/train_ACDC_10.sh      # ACDC,   10% labels
bash scripts/train_MMs2_10.sh      # M&Ms-2, 10% labels
```

Outputs are written under the experiment directory (e.g. `./outputs/ACDC_10/`):
`model/` (checkpoints: `best_seg1.pth`, `best_seg2.pth`, `best_crm.pth`),
`graph/` (TensorBoard logs) and `predictions/` (`*_Pred.nii.gz`). `best_seg1.pth`
is the primary network kept for single-network inference; `best_seg2.pth` is the
auxiliary co-training network used only when `--ensemble` is set.

Equivalent explicit command:

```bash
python train.py \
  --exp_dir ./outputs/ACDC_10 \
  --data_dir ./data/ACDC/training --test_data_dir ./data/ACDC/testing \
  --train_data_csv ./data_dir/train_ACDC.csv --valid_data_csv ./data_dir/valid_ACDC.csv \
  --test_data_list ./data_dir/test_subj_ACDC.csv \
  --mode train --label_ratio 0.1 --batch_size 4 --max_iterations 30000 \
  --lambda1 2 --lambda2 0.01 --device 0 \
  --pri --crm --crm_of --ensemble
```

Key arguments:

| Argument                      | Meaning                                                                    |
| ----------------------------- | -------------------------------------------------------------------------- |
| `--exp_dir`                 | experiment directory for checkpoints, logs and predictions                 |
| `--label_ratio`             | fraction of labeled subjects (e.g.`0.1` for 10%)                         |
| `--pri`                     | enable the PRI contrastive loss                                            |
| `--crm` / `--crm_of`      | enable CRM refinement / use refined logits as the prediction               |
| `--ensemble`                | keep the second network (Seg2) and average both at inference               |
| `--lambda1` / `--lambda2` | weights for the URC and PRI losses (default`2` / `0.01`)               |
| `--mode`                    | `train` (train + test) or `test` (test only, reuses saved checkpoints) |

## Evaluation

After inference, compute metrics from the saved predictions:

```bash
# ACDC
python calculate_metrics.py \
  --gt_dir ./data/ACDC/testing \
  --pred_dir ./outputs/ACDC_10/predictions \
  --test_csv ./data_dir/test_subj_ACDC.csv \
  --output_csv ./outputs/ACDC_10/metrics_ACDC_10.csv

# M&Ms-2
python calculate_metrics.py \
  --gt_dir ./data/MnM-2/all \
  --pred_dir ./outputs/MMs2_10/predictions \
  --test_csv ./data_dir/test_LA.csv \
  --output_csv ./outputs/MMs2_10/metrics_MMs2_10.csv
```

The output CSV lists per-subject Dice / ASSD / HD / volume scores followed by
`Mean` and `SD` rows.
