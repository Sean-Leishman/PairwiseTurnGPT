import os
import logging
import argparse

from datetime import datetime


def get_abs_path(filepath):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filepath
    )


def get_logger(namespace=__name__, level=logging.INFO):
    logger = logging.getLogger(namespace)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


logger = get_logger(__name__)


def get_configs(
    parser: argparse.ArgumentParser,
    configs: dict,
    config_namespace: argparse.Namespace,
    default: list[str] = [],
):
    result_runs = config_namespace.result_run
    original = vars(config_namespace)

    if not isinstance(result_runs, list):
        result_runs = [result_runs]

    for result_run in result_runs:
        config_run = configs.get(result_run, None)
        config_default = []
        if isinstance(config_run, dict):
            config_default = config_run.get("default", [])
            if len(config_default) == 0:
                logger.warning(f"Default config not present in {result_run}")
            config_run = config_run.get("runs", None)

        if config_run is None:
            print(f"{result_run} not present in configs of keys {config_run.keys()}")
            return []

        for config in config_run:
            config.extend(default)
            config.extend(config_default)

            new_config = argparse.Namespace(**original)
            update = parser.parse_args(args=config, namespace=new_config)

            yield update


def get_new_filename(save_dir):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(save_dir, current_time_str)


def get_latest_model(path):
    list_dir = os.listdir(path)
    latest_model = None
    latest_index = -1
    for item in list_dir:
        if item[:5] == "model":
            index = int("".join(x for x in item if x.isdigit()))
            if latest_model is None or index > latest_index:
                latest_index = index
                latest_model = item

    if latest_model is None:
        raise RuntimeError("model file not found")
    return os.path.join(path, latest_model)
