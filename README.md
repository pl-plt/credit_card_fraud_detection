# Détection de Fraude par Carte de Crédit avec PySpark

Ce projet implémente un pipeline complet de Machine Learning distribué avec **Apache Spark (PySpark)** pour détecter les transactions frauduleuses. Le dataset utilisé est fortement déséquilibré (seulement ~0.17% de fraudes), ce qui nécessite des stratégies adaptées (pondération des classes, métriques robustes comme l'AUPRC).

## Fonctionnalités

- **Preprocessing Scalable** : Conversion CSV vers Parquet, typage, et split train/test stratifié.
- **Multi-Modèles** : Entraînement et comparaison de :
  - Logistic Regression (avec gestion du déséquilibre via `weightCol`).
  - Random Forest Classifier.
  - Gradient Boosted Trees (GBT).
- **Pipeline ML** : Utilisation de `Pipeline` Spark MLlib (VectorAssembler, StandardScaler, Estimateurs).
- **Scoring Batch** : Script dédié pour l'inférence sur de nouveaux datasets.
- **Visualisation** : Notebooks pour l'analyse exploratoire et les courbes de performance (ROC, PR).

## Structure du Projet

```
.
├── dataset/
│   ├── raw/                  # Fichier creditcard.csv original
│   └── processed/            # Données converties en Parquet (train/test)
├── models/                   # Pipelines entraînés et métriques JSON
├── notebooks/
│   ├── cc_fraud_spark.ipynb       # Pipeline interactif complet (Dev)
│   └── model_comparison_viz.ipynb # Visualisation des résultats et comparaison
├── src/
│   ├── preprocess.py         # Nettoyage et préparation des données
│   ├── train.py              # Entraînement des modèles (LR, RF, GBT)
│   └── score.py              # Scoring et évaluation batch
├── run.sh                    # Script d'exécution complet (Demo)
├── requirements.txt          # Dépendances Python
└── README.md
```

## Installation et Prérequis

### Prérequis
- **Java 8, 11 ou 17** (Requis pour Spark).
- **Python 3.8+** (Recommandé via Conda).
- **Hadoop (Windows uniquement)** : `winutils.exe` et `hadoop.dll` doivent être configurés.

### Installation

1. **Cloner le dépôt**
2. **Créer l'environnement virtuel**
   ```bash
   conda create -n spark python=3.13 -y
   conda activate spark
   pip install -r requirements.txt
   ```
3. **Configuration Windows (Important)**
   Définissez la variable d'environnement `HADOOP_HOME` vers votre dossier contenant `bin/winutils.exe` (ex: `C:\hadoop-3.3.6`).

## Dépendances Principales

Basé sur `requirements.txt` :
- **PySpark** 4.0.1
- **Scikit-learn** 1.7.1
- **Pandas** 2.3.3
- **Matplotlib** 3.10.7
- **Kagglehub** 0.3.13

## Utilisation

### 1. Mode Automatique
Le script `run.sh` exécute tout le pipeline : préparation, entraînement des 3 modèles et scoring avec le meilleur modèle.

```bash
bash run.sh
```

### 2. Mode Manuel (Étape par étape)
Si jamais vous souhaitez exécuter chaque étape séparément voici les commandes a utiliser :


**A. Preprocessing**
Convertit le CSV en Parquet et sépare les jeux de données.
```bash
python src/preprocess.py --input dataset/raw/creditcard.csv --output dataset/processed --train-ratio 0.8
```

**B. Entraînement**
Entraîner différents modèles. Les modèles sont sauvegardés dans `models/`.

*Logistic Regression :*
```bash
python src/train.py --train dataset/processed/train --test dataset/processed/test --model-dir models/lr_pipeline --model-type lr
```

*Random Forest :*
```bash
python src/train.py --train dataset/processed/train --test dataset/processed/test --model-dir models/rf_pipeline --metrics models/metrics_rf.json --model-type rf
```

*Gradient Boosted Trees (Meilleur modèle) :*
```bash
python src/train.py --train dataset/processed/train --test dataset/processed/test --model-dir models/gbt_pipeline --model-type gbt
```

**C. Scoring**
Appliquer le modèle sur le jeu de test et générer des scores de probabilité.
```bash
python src/score.py --model-dir models/gbt_pipeline --input dataset/processed/test --output dataset/processed/scores_gbt --threshold 0.5
```

## Résultats et Performance

Les modèles sont évalués principalement sur l'**Area Under Precision-Recall Curve (AUPRC)**, métrique plus adaptée que l'AUC-ROC pour les datasets très déséquilibrés.

| Modèle | AUPRC | AUROC | Temps d'entraînement |
|--------|-------|-------|----------------------|
| **Logistic Regression** | ~0.68 | ~0.97 | Rapide |
| **Random Forest** | ~0.83 | ~0.97 | Moyen |
| **Gradient Boosted Trees** | **~0.85** | **~0.99** | Lent |

> *Note : Les résultats peuvent varier légèrement selon le seed et le split des données.*

## Notebooks

- **`cc_fraud_spark.ipynb`** : Notebook principal pour le développement. Contient le chargement, l'exploration, l'entraînement des variantes et l'export du meilleur modèle.
- **`model_comparison_viz.ipynb`** : Charge les modèles entraînés et les prédictions pour générer des graphiques avancés (Confusion Matrix, Precision-Recall Curve) et ajuster le seuil de décision optimal.