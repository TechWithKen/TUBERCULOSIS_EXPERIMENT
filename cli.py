import argparse

from TUBERCULOSIS_EXPERIMENT.main import (
    run_training_pipeline
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["train"]
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline()