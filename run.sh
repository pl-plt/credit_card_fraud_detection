#!/bin/bash

# 1) Préprocess: lecture CSV -> parquet + split train/test
echo -e "\033[1;34m========== Preprocessing des donnees ==========\033[0m"
python src/preprocess.py --input dataset/raw/creditcard.csv --output dataset/processed --train-ratio 0.8 --seed 42

# 2) Entraînement. GBT = meilleur résultat dans nos tests.
# Logistic Regression
echo -e "\033[1;34m========== Entrainement selon un modele de regression logistique ==========\033[0m"
python src/train.py --train dataset/processed/train --test dataset/processed/test --model-dir models/lr_pipeline --metrics models/metrics.json --model-type lr

# RandomForest
echo -e "\033[1;34m========== Entrainement selon un modele Random Forest ==========\033[0m"
python src/train.py --train dataset/processed/train --test dataset/processed/test --model-dir models/rf_pipeline --metrics models/metrics_rf.json --model-type rf

# Gradient Boosted Trees
echo -e "\033[1;34m========== Entrainement selon un modele Gradient Boosted Trees ==========\033[0m"
python src/train.py --train dataset/processed/train --test dataset/processed/test --model-dir models/gbt_pipeline --metrics models/metrics_gbt.json --model-type gbt

# 3) Scoring batch (adapter le modèle/threshold si besoin)
echo -e "\033[1;34m========== Scoring batch ==========\033[0m"
python src/score.py --model-dir models/gbt_pipeline --input dataset/processed/test --output dataset/processed/scores_gbt --threshold 0.5