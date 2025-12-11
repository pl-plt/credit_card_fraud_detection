import argparse
import json
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


def build_spark(app_name: str = "cc-fraud-train") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_parquet(spark: SparkSession, path: str):
    return spark.read.parquet(path)


def add_class_weights(df, label_col: str = "Class", weight_col: str = "classWeightCol"):
    counts = df.groupBy(label_col).count().collect()
    stats = {row[label_col]: row["count"] for row in counts}
    total = sum(stats.values())
    pos = stats.get(1, 1)
    balancing_ratio = (total - pos) / total
    df = df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1, balancing_ratio).otherwise(1 - balancing_ratio),
    )
    return df, stats


def build_pipeline(feature_cols, model_type: str = "lr"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
    evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderPR")

    if model_type == "lr":
        estimator = LogisticRegression(featuresCol="features", labelCol="Class", weightCol="classWeightCol")
        param_grid = (
            ParamGridBuilder()
            .addGrid(estimator.regParam, [0.01, 0.1])
            .addGrid(estimator.elasticNetParam, [0.0, 0.5])
            .build()
        )
        stages = [assembler, scaler, estimator]
    elif model_type == "rf":
        estimator = RandomForestClassifier(
            featuresCol="features",
            labelCol="Class",
            weightCol="classWeightCol",
            numTrees=200,
            maxDepth=10,
            subsamplingRate=0.8,
        )
        param_grid = (
            ParamGridBuilder()
            .addGrid(estimator.maxDepth, [8, 12])
            .addGrid(estimator.numTrees, [100, 200])
            .build()
        )
        stages = [assembler, scaler, estimator]
    elif model_type == "gbt":
        estimator = GBTClassifier(
            featuresCol="features",
            labelCol="Class",
            maxDepth=5,
            maxIter=50,
            subsamplingRate=0.8,
        )
        param_grid = (
            ParamGridBuilder()
            .addGrid(estimator.maxIter, [30, 60])
            .addGrid(estimator.maxDepth, [4, 6])
            .build()
        )
        stages = [assembler, scaler, estimator]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(stages=stages)

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2,
    )
    return tvs, evaluator


def evaluate(preds, label_col: str = "Class"):
    eval_pr = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR")
    eval_roc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
    return {
        "auprc": eval_pr.evaluate(preds),
        "auroc": eval_roc.evaluate(preds),
    }


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Spark ML pipeline for fraud detection")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_train = os.path.normpath(os.path.join(base_dir, "..", "dataset", "processed", "train"))
    default_test = os.path.normpath(os.path.join(base_dir, "..", "dataset", "processed", "test"))
    default_model = os.path.normpath(os.path.join(base_dir, "..", "models", "lr_pipeline"))
    default_metrics = os.path.normpath(os.path.join(base_dir, "..", "models", "metrics.json"))

    parser.add_argument("--train", default=default_train, help="Path to train parquet")
    parser.add_argument("--test", default=default_test, help="Path to test parquet")
    parser.add_argument("--model-dir", default=default_model, help="Output dir for model")
    parser.add_argument("--metrics", default=default_metrics, help="Path to metrics JSON")
    parser.add_argument("--model-type", choices=["lr", "rf", "gbt"], default="lr", help="Model family to train")
    args = parser.parse_args()

    spark = build_spark()

    train_df = load_parquet(spark, args.train)
    test_df = load_parquet(spark, args.test)

    train_df, train_stats = add_class_weights(train_df)
    test_df = test_df.withColumn("classWeightCol", F.lit(1.0))

    feature_cols = [c for c in train_df.columns if c not in {"Class", "classWeightCol"}]

    tvs, evaluator = build_pipeline(feature_cols, args.model_type)
    tv_model = tvs.fit(train_df)

    preds = tv_model.transform(test_df)
    metrics = evaluate(preds)
    metrics["train_counts"] = train_stats

    os.makedirs(args.model_dir, exist_ok=True)
    tv_model.bestModel.write().overwrite().save(args.model_dir)
    save_metrics(metrics, args.metrics)

    print("\033[92m" + json.dumps(metrics, indent=2) + "\033[0m")

    spark.stop()


if __name__ == "__main__":
    main()
