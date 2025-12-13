#!/bin/bash

# 0) Préparer l'environnement (une seule fois)
echo -e "\033[1;34m========== Preparation de l'environnement ==========\033[0m"
conda init
if ! conda env list | grep -qw spark; then
    conda create -n spark python=3.13 -y
fi
conda activate spark
pip install -r requirements.txt

# 1) Téléchargement du dataset (si nécessaire)
echo -e "\033[1;34m========== Telechargement du dataset ==========\033[0m"
python src/import_dataset.py --output dataset/raw
