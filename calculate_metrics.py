"""Compute per-subject and mean DSC / ASSD / HD / volume metrics.

Reads the ``*_Pred.nii.gz`` volumes written by ``train.py`` (test mode) and the
ground-truth labels, and writes a CSV of per-subject scores plus Mean / SD rows.

Examples
--------
ACDC (10% run):
    python calculate_metrics.py \
        --gt_dir ./data/ACDC/testing \
        --pred_dir ./outputs/ACDC_10/predictions \
        --test_csv ./data_dir/test_subj_ACDC.csv \
        --output_csv ./outputs/ACDC_10/metrics_ACDC_10.csv

M&Ms-2 (10% run):
    python calculate_metrics.py \
        --gt_dir ./data/MnM-2/all \
        --pred_dir ./outputs/MMs2_10/predictions \
        --test_csv ./data_dir/test_LA.csv \
        --output_csv ./outputs/MMs2_10/metrics_MMs2_10.csv
"""
import os
import argparse

import numpy as np
import nibabel as nib
import pandas as pd

from utilities import image_utils


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PRICoD predictions")
    p.add_argument("--gt_dir", type=str, default="./data/ACDC/testing",
                   help="Root directory of the ground-truth label volumes")
    p.add_argument("--pred_dir", type=str, default="./outputs/ACDC_10/predictions",
                   help="Directory of the predicted '*_Pred.nii.gz' volumes")
    p.add_argument("--test_csv", type=str, default="./data_dir/test_subj_ACDC.csv",
                   help="CSV listing the test subjects")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--output_csv", type=str, default="./outputs/metrics.csv")
    return p.parse_args()


def main():
    args = parse_args()
    num_classes = args.num_classes

    label_names = pd.read_csv(args.test_csv)["label_filenames"].tolist()

    dice_cols = [f"DiceClass{i}" for i in range(1, num_classes)]
    assd_cols = [f"AssdClass{i}" for i in range(1, num_classes)]
    hd_cols = [f"HdClass{i}" for i in range(1, num_classes)]
    volE_cols = [f"VolEClass{i}" for i in range(1, num_classes)]
    volP_cols = [f"VolPClass{i}" for i in range(1, num_classes)]
    volPred_cols = [f"VolPredClass{i}" for i in range(1, num_classes)]
    volGT_cols = [f"VolGTClass{i}" for i in range(1, num_classes)]

    rows = []
    for label_name in label_names:
        gt_path = os.path.join(args.gt_dir, label_name)
        pred_path = os.path.join(args.pred_dir, label_name.replace("_gt.nii.gz", "_Pred.nii.gz"))
        print(f"GT: {gt_path}  |  Pred: {pred_path}")

        gt_nii = nib.load(gt_path)
        pixel_spacing = gt_nii.header["pixdim"][1:4]
        gt = gt_nii.get_fdata().astype(np.int8)
        pred = nib.load(pred_path).get_fdata().astype(np.int8)

        dice = image_utils.np_categorical_dice_3d(pred, gt, num_classes)
        assd, hd = image_utils.np_categorical_assd_hd(pred, gt, num_classes, pixel_spacing[0:2])
        volE, volP, vol_pred, vol_gt = image_utils.np_categorical_volume_3d(
            pred, gt, num_classes, pixel_spacing[0:3])

        row = {"Subject": label_name}
        for j in range(1, num_classes):
            row[f"DiceClass{j}"] = dice[j]
            row[f"AssdClass{j}"] = assd[j]
            row[f"HdClass{j}"] = hd[j]
            row[f"VolEClass{j}"] = volE[j]
            row[f"VolPClass{j}"] = volP[j]
            row[f"VolPredClass{j}"] = vol_pred[j]
            row[f"VolGTClass{j}"] = vol_gt[j]
        rows.append(row)

    cols = (["Subject"] + dice_cols + assd_cols + hd_cols
            + volE_cols + volP_cols + volPred_cols + volGT_cols)
    df = pd.DataFrame(rows, columns=cols)

    summary = {}
    for name, fn in [("Mean", lambda c: c.mean(skipna=True)),
                     ("SD", lambda c: c.std(ddof=0, skipna=True)),
                     ("Max", lambda c: c.max(skipna=True)),
                     ("Min", lambda c: c.min(skipna=True))]:
        s = fn(df.drop(columns=["Subject"]))
        s["Subject"] = name
        summary[name] = s
    df = pd.concat([df, pd.DataFrame(list(summary.values()))], ignore_index=True)
    df = df.round(4)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False, na_rep="nan")
    print(f"\nSaved metrics to {args.output_csv}")
    print(df[df["Subject"].isin(["Mean", "SD"])][["Subject"] + dice_cols + assd_cols].to_string(index=False))


if __name__ == "__main__":
    main()
