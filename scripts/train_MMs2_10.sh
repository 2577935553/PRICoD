#!/bin/bash
# PRICoD on M&Ms-2 (long-axis) with a 10% labeling ratio (full model: PRI + CRM + URC).
python train.py \
  --exp_dir         ./outputs/MMs2_10 \
  --data_dir        ./data/MnM-2/all \
  --train_data_csv  ./data_dir/train_LA.csv \
  --valid_data_csv  ./data_dir/valid_LA.csv \
  --test_data_dir   ./data/MnM-2/all \
  --test_data_list  ./data_dir/test_LA.csv \
  --mode train \
  --label_ratio 0.1 \
  --batch_size 4 \
  --lambda1 2 \
  --lambda2 0.01 \
  --max_iterations 30000 \
  --device 0 \
  --pri --crm --crm_of --ensemble

# Evaluate:
# python calculate_metrics.py \
#   --gt_dir ./data/MnM-2/all \
#   --pred_dir ./outputs/MMs2_10/predictions \
#   --test_csv ./data_dir/test_LA.csv \
#   --output_csv ./outputs/MMs2_10/metrics_MMs2_10.csv
