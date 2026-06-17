from pathlib import Path

from src.regression import run_regression_experiment


DATA_PATH = Path("data/house_pricing.csv")
OUTPUT_DIR = Path("outputs")


def main():
    run_regression_experiment(DATA_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
