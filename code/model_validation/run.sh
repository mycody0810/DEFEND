#!/bin/bash

# Define parameters
K_RANDOM_STATE=1
ALL_COUNT=20000
K_FOLD=5
OVERSAMPLE=1

RANDOM_STATE_list=(0 1)
MODEL_NAME_list=("RFC" "LR" "KNN" "Efficient" "MLP" "Autoencoder")  # Replace with your specific model names
FEATURE_VERSION_list=("feature7/all")
# Set to your project root directory
export PYTHONPATH="D:\Work\research\netflow\code\Deep-Feature-for-Network-Threat-Detection"

# Run Python script
for model_name in "${MODEL_NAME_list[@]}"; do
# for all_count in "${ALL_COUNT_list[@]}"; do
  for random_state in "${RANDOM_STATE_list[@]}"; do
    for feature_version in "${FEATURE_VERSION_list[@]}"; do
      # Call Python script and pass in current item
      "python.exe" ".\Deep-Feature-for-Network-Threat-Detection\run.py" --kfold_random_state "$K_RANDOM_STATE" --random_state "$random_state" --all_count "$ALL_COUNT" --k_fold "$K_FOLD" --model_name "$model_name" --feature_version "$feature_version" --oversample_all "$OVERSAMPLE"
    done
  done
done