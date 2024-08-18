import os
import json

from argparse import ArgumentParser, Namespace


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--evaluate-on-full", action="store_true")
    parser.add_argument("--evaluate-on-serialised", action="store_true")
    parser.add_argument("--remove-overlaps", action="store_true")
    parser.add_argument("--remove-backchannels", action="store_true")

    return parser


def list_models(filter):
    for file in os.listdir("trained_model"):
        file = os.path.join("trained_model", file)
        if not os.path.isdir(file):
            print(f"{file} is not a directory")
            continue

        config_file = os.path.join(file, "config.json")
        if not os.path.exists(config_file):
            print("No config file found for ", file)
            continue

        f_model = None
        for model_file in os.listdir(file):
            if model_file[-3:] == ".pt":
                f_model = os.path.join(file, model_file)
                break

        if f_model is None:
            print("No model found for ", file)
            continue

        with open(config_file) as f:
            config = Namespace(**json.load(f))

            if filter.evaluate_on_full and not config.evaluate_on_full:
                continue
            if filter.evaluate_on_serialised and not config.evaluate_on_serialised:
                continue
            if filter.remove_overlaps and not config.remove_overlaps:
                continue
            if filter.remove_backchannels and not config.remove_backchannels:
                continue

            print(f"{file}: {config.run_name}:{config.detail}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    list_models(args)
