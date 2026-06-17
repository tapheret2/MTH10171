from pathlib import Path

from src.classification import run_classification_experiment


DATA_PATH = Path("data/loan_approval_dataset.csv")
OUTPUT_DIR = Path("outputs")


def main():
    run_classification_experiment(DATA_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()

