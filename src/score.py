import argparse
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


def build_spark(app_name: str = "cc-fraud-score") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def main():
    parser = argparse.ArgumentParser(description="Score new data with saved Spark model")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.normpath(os.path.join(base_dir, "..", "models", "lr_pipeline"))
    default_input = os.path.normpath(os.path.join(base_dir, "..", "dataset", "processed", "test"))
    default_output = os.path.normpath(os.path.join(base_dir, "..", "dataset", "processed", "scores"))

    parser.add_argument("--model-dir", default=default_model, help="Path to saved PipelineModel")
    parser.add_argument("--input", default=default_input, help="Input parquet data to score")
    parser.add_argument("--output", default=default_output, help="Output parquet for scores")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary prediction")
    args = parser.parse_args()

    spark = build_spark()
    model = PipelineModel.load(args.model_dir)

    df = spark.read.parquet(args.input)
    scored = model.transform(df)

    # Extract probability of class 1 via vector_to_array (avoids UDT issues)
    scored = scored.withColumn("prob_array", vector_to_array(F.col("probability")))
    scored = scored.withColumn("score", F.col("prob_array")[1])
    scored = scored.drop("prediction_thresh")
    scored = scored.withColumn("prediction_thresh", (F.col("score") >= args.threshold).cast("double"))
    # Nettoyages des colonnes intermédiaires pour écrire un schéma épuré
    scored = scored.drop("prob_array")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scored.write.mode("overwrite").parquet(args.output)

    # Quick metrics if labels are present
    if "Class" in scored.columns:
        tp = scored.filter((F.col("prediction_thresh") == 1) & (F.col("Class") == 1)).count()
        fp = scored.filter((F.col("prediction_thresh") == 1) & (F.col("Class") == 0)).count()
        fn = scored.filter((F.col("prediction_thresh") == 0) & (F.col("Class") == 1)).count()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print("\033[92m" + str(json.dumps({"precision": precision, "recall": recall, "f1": f1}, indent=2)) + "\033[0m")

    spark.stop()


if __name__ == "__main__":
    main()
