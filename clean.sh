#!/bin/bash

# Nettoyage des dossiers de données et modèles
echo -e "\033[1;34m========== Nettoyage des dossiers de donnees et modeles ==========\033[0m"
echo "Removing dataset/processed/"
rm -rf dataset/processed/*
echo "Removing dataset/raw/"
rm -rf dataset/raw/*
echo "Removing models/"
rm -rf models/*
echo -e "\033[0;32mDone!\033[0m"
