import argparse
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType


def build_spark(app_name: str = "cc-fraud-preprocess") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def read_csv(spark: SparkSession, path: str):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )


def clean_df(df):
    for col, dtype in df.dtypes:
        if dtype in {"int", "bigint", "float", "double"} and col != "Class":
            df = df.withColumn(col, F.col(col).cast(DoubleType())) 
    df = df.dropDuplicates()
    return df


def write_parquet(df, path: str):
    df.write.mode("overwrite").parquet(path)


def stratified_split(df, train_ratio: float, seed: int = 42):
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    return train_df, test_df


def compute_class_stats(df):
    n_rows = df.count()
    counts = df.groupBy("Class").count().collect()
    stats = {row["Class"]: row["count"] for row in counts}
    return n_rows, stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess credit card fraud dataset with Spark")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.normpath(os.path.join(base_dir, "..", "dataset", "raw", "creditcard.csv"))
    default_output = os.path.normpath(os.path.join(base_dir, "..", "dataset", "processed"))

    parser.add_argument("--input", default=default_input, help="Path to input CSV")
    parser.add_argument("--output", default=default_output, help="Base output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    spark = build_spark()
    df = read_csv(spark, args.input)
    n_rows, class_stats = compute_class_stats(df)
    print(f"Rows: {n_rows}, class distribution: {class_stats}")

    df = clean_df(df)

    os.makedirs(args.output, exist_ok=True)
    raw_parquet = os.path.join(args.output, "raw_parquet")
    write_parquet(df, raw_parquet)

    train_df, test_df = stratified_split(df, args.train_ratio, args.seed)
    train_path = os.path.join(args.output, "train")
    test_path = os.path.join(args.output, "test")
    write_parquet(train_df, train_path)
    write_parquet(test_df, test_path)

    train_rows, train_stats = compute_class_stats(train_df)
    test_rows, test_stats = compute_class_stats(test_df)
    print(f"Train rows: {train_rows}, class distribution: {train_stats}")
    print(f"Test rows: {test_rows}, class distribution: {test_stats}")

    spark.stop()


if __name__ == "__main__":
    main()
