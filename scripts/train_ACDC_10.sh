#!/bin/bash
# PRICoD on ACDC with a 10% labeling ratio (full model: PRI + CRM + URC).
# Trains Seg1 and Seg2, then runs inference and writes predictions.
python train.py \
  --exp_dir         ./outputs/ACDC_10 \
  --data_dir        ./data/ACDC/training \
  --train_data_csv  ./data_dir/train_ACDC.csv \
  --valid_data_csv  ./data_dir/valid_ACDC.csv \
  --test_data_dir   ./data/ACDC/testing \
  --test_data_list  ./data_dir/test_subj_ACDC.csv \
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
#   --gt_dir ./data/ACDC/testing \
#   --pred_dir ./outputs/ACDC_10/predictions \
#   --test_csv ./data_dir/test_subj_ACDC.csv \
#   --output_csv ./outputs/ACDC_10/metrics_ACDC_10.csv
