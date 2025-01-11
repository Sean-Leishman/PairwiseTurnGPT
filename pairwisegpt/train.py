import random

import torch
import transformers
from torch.utils.data import DataLoader

import argparse
import logging
import json
import os
import warnings

import wandb

from data.dialog_dm import DialogDM
from data.serialised_process import SerialisedProcessType
from types import SimpleNamespace

from pairwisegpt.build_metrics import BuildMetrics
from pairwisegpt.model import PairwiseGPT, DefaultModel
from pairwisegpt.trainer import PairwiseTrainer as Trainer
from pairwisegpt.utils import get_abs_path, get_latest_model, get_logger


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logger = get_logger(__name__, level=logging.DEBUG)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bespoke Tart model used to predict turn-taking from linguistic features"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="true/false if cuda should be enabled"
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--load-model", action="store_true", help="true/false if model should be loaded"
    )
    model_group.add_argument(
        "--load-path",
        type=str,
        default="trained_model/",
        help="load model config and weights from this file and ignore input configurations",
    )
    model_group.add_argument(
        "--save-path",
        type=str,
        default="trained_model/",
        help="model weights and config options save directory",
    )

    model_group.add_argument(
        "--finetune", action="store_true", help="true/false if BERT should be finetuned"
    )
    model_group.add_argument(
        "--pretrained", type=str, default="gpt2", help="name of pretrained BERT model"
    )
    model_group.add_argument("--save-model-allowed", action="store_true")

    model_group.add_argument(
        "--lora", action="store_true", help="enable lora training of model"
    )

    trainer_group = parser.add_argument_group("trainer")
    trainer_group.add_argument("--epochs", type=int, default=10)
    trainer_group.add_argument("--batch-size", type=int, default=128)
    trainer_group.add_argument("--learning-rate", type=float, default=0.0000625)
    trainer_group.add_argument(
        "--pretrained-learning-rate", type=float, default=0.0000625
    )
    trainer_group.add_argument("--weight-decay", type=float, default=0.05)
    trainer_group.add_argument(
        "--early-stop",
        type=int,
        default=5,
        help="number of iterations without improvement for early stop",
    )
    trainer_group.add_argument("--weight-eos-token", type=float, default=1)
    trainer_group.add_argument("--weight-reg-token", type=float, default=0.5)
    trainer_group.add_argument("--weight-tokens", action="store_true")
    trainer_group.add_argument("--remove-cross-attention", action="store_true")

    trainer_group.add_argument(
        "--remove-emp-metric-generation",
        action="store_true",
        help="[REMOVED FUNCTIONALITY] remove emp metric generation",
    )

    trainer_group.add_argument(
        "--evaluate",
        action="store_true",
        help="model should only be evaluated. load-model and load-path should be set",
    )

    trainer_group.add_argument(
        "--description", type=str, default="", help="description of model"
    )

    trainer_group.add_argument(
        "--run-name", type=str, default="", help="set name of run"
    )
    trainer_group.add_argument(
        "--result-run", type=str, nargs="+", default="", help="run multiple configs"
    )
    trainer_group.add_argument(
        "--detail", type=str, default="", help="append to --run-name"
    )

    # Wandb
    trainer_group.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="frequency with which to report logs to wandb",
    )

    trainer_group.add_argument("--evaluate-on-full", action="store_true")
    trainer_group.add_argument("--evaluate-on-serialised", action="store_true")

    # Dataset
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--overwrite", action="store_true", help="overwrite and regenerate dataset"
    )
    dataset_group.add_argument(
        "--dev-mode",
        action="store_true",
        help="decrease dataset size to test post-processing steps/disable writing to wandb",
    )

    dataset_group.add_argument(
        "--use-default-model",
        action="store_true",
        help="Default model only outputs 0s/1s",
    )

    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        type=str,
        help="[DEPRECATED: use `--train-datasets` and `--eval-datasets`] datasets to use",
    )
    dataset_group.add_argument(
        "--train-datasets",
        nargs="+",
        help="Datasets to train the model using",
        default=[
            "switchboard",
        ],
        choices=["switchboard", "fisher", "edacc"],
    )
    dataset_group.add_argument(
        "--use-test-dataset-for-val",
        action="store_true",
        help="Use test dataset for validation. If not set, use train dataset for validation",
    )
    dataset_group.add_argument(
        "--eval-datasets",
        nargs="+",
        help="Datasets to evaluate the model over",
        default=[
            "switchboard",
        ],
        choices=["switchboard", "fisher", "edacc"],
    )

    dataset_group.add_argument(
        "--max-length", type=int, default=256, help="max length of a sequence"
    )
    dataset_group.add_argument(
        "--keep-length", type=int, default=64, help="minimum length of a sequence"
    )
    dataset_group.add_argument(
        "--overlap-length",
        type=int,
        default=10,
        help="number of tokens to overlap between sequences",
    )
    dataset_group.add_argument(
        "--yield-overlap-thresh",
        type=float,
        default=2,
        help="number of seconds which overlap is from turn end to define turn end as yield",
    )

    dataset_group.add_argument(
        "--include-speaker-embeddings",
        action="store_true",
        help="add speaker tokens as token type ids",
    )

    dataset_group.add_argument(
        "--include-yield-token",
        action="store_true",
        help="[DEPRECATED: use `--end-of-utterance-tokens`] add yield token to end of utterance",
    )
    dataset_group.add_argument(
        "--include-end-bc-token",
        action="store_true",
        help="[DEPRECATED: use `--end-of-utterance-tokens`] add ebc token to end of utterance",
    )
    dataset_group.add_argument(
        "--include-overlap-token",
        action="store_true",
        help="[DEPRECATED: use `--end-of-utterance-tokens`] add eint token to end of utterance",
    )
    dataset_group.add_argument(
        "--end-of-utterance-tokens",
        nargs="+",
        default=["<eint>", "<ebc>", "<yield>"],
        help="tokens to include for end of utterance [<eint>, <ebc>,  <yield>]",
    )

    dataset_group.add_argument(
        "--remove-overlaps",
        action="store_true",
        help="[DEPRECATED: use `--include-overlaps`] remove overlaps from data when parsing",
    )
    dataset_group.add_argument(
        "--include-overlaps",
        action="store_true",
        help="include overlaps from data when parsing",
    )

    dataset_group.add_argument(
        "--remove-backchannels",
        action="store_true",
        help="[DEPRECATED: use `--include-overlaps`] remove backchannels from data when parsing",
    )
    dataset_group.add_argument(
        "--include-backchannels",
        action="store_true",
        help="include backchannels from data when parsing",
    )
    dataset_group.add_argument(
        "--include-partial-overlaps",
        action="store_true",
        help="include partial overlaps from data when parsing",
    )

    dataset_group.add_argument(
        "--remove-start-tokens",
        action="store_true",
        help="[REMOVED FUNCTIONALITY] remove start tokens from the dataset",
    )
    dataset_group.add_argument(
        "--serialise-data",
        action="store_true",
        help="perform same perprocessing done for gptonly except pairwise. Note, automatically sets `--remove-start-tokens`",
    )

    dataset_group.add_argument("--single-stream", action="store_true")

    dataset_group.add_argument(
        "--include-bc-token",
        action="store_true",
        help="replace all backchannel words with specific <bc> token",
    )

    dataset_group.add_argument(
        "--filter-bc-overlap-token",
        action="store_true",
        help="[DEPRECATED: use `--filter-special-tokens` with `eint` and `ebc`] filter out backchannel and overlap tokens from the dataset",
    )
    dataset_group.add_argument(
        "--no-emp-tokens",
        action="store_true",
        help="[DEPRECATED: use `--filter-special-tokens` with `emp`] filter out <emp> tokens from the dataset",
    )
    dataset_group.add_argument(
        "--filter-special-tokens",
        nargs="+",
        default=[],
        help="filter out these tokens from the dataset. replaces `--no-emp-tokens` and `--filter-bc-overlap-token`",
    )

    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--random-seed", type=int, default=-1)

    return parser


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def load_ds(config, model, split="val", end_of_utterance_tokens=["<ebc>", "<eint>"]):
    if split == "train":
        datasets = config.train_datasets
    elif split == "val" and config.use_test_dataset_for_val:
        datasets = config.eval_datasets
    elif split == "val" and not config.use_test_dataset_for_val:
        datasets = config.train_datasets
    elif split == "test":
        datasets = config.eval_datasets
    else:
        raise ValueError(f"Invalid split {split}")

    if config.evaluate_on_full and not config.evaluate_on_serialised:
        ds = DialogDM(
            split=split,
            tokenizer=model.get_tokenizer(),
            load_from_cache=not config.overwrite,
            include_overlaps=True,
            include_backchannels=True,
            include_partial_overlaps=True,
            include_bc_token=config.include_bc_token,
            method="aligned",
            end_of_utterance_tokens=end_of_utterance_tokens,
            keep_in_memory=False,
            filter_special_tokens=config.filter_special_tokens,
            max_length=config.max_length,
            keep_length=config.keep_length,
            overlap_length=config.overlap_length,
            datasets=datasets,
        )
        ds.prepare_data()
    elif config.evaluate_on_serialised:
        ds = DialogDM(
            split=split,
            tokenizer=model.get_tokenizer(),
            load_from_cache=not config.overwrite,
            serialised=SerialisedProcessType.TurnEnd,
            method="serialised",
            combine_speaker=False,  # Keep two speaker streams
            end_of_utterance_tokens=end_of_utterance_tokens,
            keep_in_memory=False,
            include_bc_token=config.include_bc_token,
            filter_special_tokens=config.filter_special_tokens,
            max_length=config.max_length,
            keep_length=config.keep_length,
            overlap_length=config.overlap_length,
            datasets=datasets,
        )
        ds.prepare_data()
    else:
        config.eval_datasets = datasets
        ds = DialogDM(
            split=split,
            tokenizer=model.get_tokenizer(),
            serialised=SerialisedProcessType.TurnEnd if config.serialise_data else None,
            method="serialised" if config.serialise_data else "aligned",
            combine_speaker=False,  # Keep two speaker streams
            load_from_cache=not config.overwrite,
            keep_in_memory=False,
            **vars(config),
        )
        ds.prepare_data()

    return ds


def load_dl(ds, batch_size=4):
    dl = DataLoader(
        ds, batch_size=batch_size, collate_fn=ds.collate_fn, num_workers=8, shuffle=True
    )

    return dl


def main(config):
    config.device = "cuda" if torch.cuda.is_available() and (config.cuda) else "cpu"
    logging.getLogger(__name__).info(f"{config}")

    logging.getLogger(__name__).info(
        f"model: initialising model with name {config.run_name}"
    )

    if config.run_name == "":
        name = input("Enter new change(s) for wandb run: ")
        if name == "":
            name = None
        config.run_name = name

    wandb.init(
        config=config,
        name=f"{config.run_name}: {config.detail}",
        entity="leishy333",
        mode="online" if not config.offline else "offline",
    )
    logging.getLogger(__name__).info(f"model: set config {config}")
    logging.getLogger(__name__).info(f"model: set run name {config.run_name}")
    logging.getLogger(__name__).info(f"model: set device {config.device}")

    if config.random_seed != -1:
        logging.getLogger(__name__).info(f"model set random seed {config.random_seed}")
        set_seed(config.random_seed)

    if config.load_model:
        load_path = get_abs_path(config.load_path)
        logging.getLogger(__name__).info(f"model: loading model from {load_path}")

        with open(os.path.join(load_path, "config.json")) as f:
            new_config = SimpleNamespace(**json.load(f))
            for arg in vars(new_config):
                config.arg = getattr(new_config, arg)

        logging.getLogger(__name__).info(f"Loaded config: {config}")

    logging.getLogger(__name__).info(
        f"Loaded model: gpt with finetuning: {config.finetune}"
    )

    if config.use_default_model:
        model = DefaultModel(**vars(config))
    else:
        model = PairwiseGPT(**vars(config))

    model.to(config.device)

    new_params = []
    pretrained_params = []
    for name, param in model.named_parameters():
        if "crossattention" in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    # Set different learning rates for different parameter groups
    optimizer = torch.optim.AdamW(
        [
            {
                "params": pretrained_params,
                "lr": config.learning_rate,
                "weight_decay": 1e-5,
            },
            {"params": new_params, "lr": config.learning_rate, "weight_decay": 1e-4},
        ]
    )

    if config.load_model:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            metric_builder=BuildMetrics,
            load_from_checkpoint=get_latest_model(load_path),
            **vars(config),
        )
    else:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            metric_builder=BuildMetrics,
            **vars(config),
        )

    if not config.evaluate:
        config.datasets = config.train_datasets
        train_ds = DialogDM(
            split="train",
            tokenizer=model.get_tokenizer(),
            serialised=SerialisedProcessType.TurnEnd if config.serialise_data else None,
            method="serialised" if config.serialise_data else "aligned",
            combine_speaker=False,  # Keep two speaker streams
            load_from_cache=not config.overwrite,
            keep_in_memory=False,
            **vars(config),
        )
        train_ds.prepare_data()

        val_ds = load_ds(
            config,
            model,
            split="val",
            end_of_utterance_tokens=config.end_of_utterance_tokens,
        )
        test_ds = load_ds(
            config,
            model,
            split="test",
            end_of_utterance_tokens=config.end_of_utterance_tokens,
        )

        logging.getLogger(__name__).info("model: train model")

        scheduler = None
        history = trainer.train(
            train_ds, val_ds, test_ds, scheduler=scheduler, dataset_loader=load_dl
        )
        wandb.finish()
    else:
        val_ds = load_ds(
            config,
            model,
            split="val",
            end_of_utterance_tokens=config.end_of_utterance_tokens,
        )
        test_ds = load_ds(
            config,
            model,
            split="test",
            end_of_utterance_tokens=config.end_of_utterance_tokens,
        )

        logging.getLogger(__name__).info("model: evaluate model")
        if not config.load_model:
            logging.getLogger(__name__).error("model: model is not being loaded")
            return None

        history = trainer.evaluate(val_ds, test_ds, dataset_loader=load_dl)
        wandb.finish()

    return history


def validate_args(config):
    if config.remove_overlaps:
        if config.include_overlaps:
            logging.getLogger(__name__).error(
                "Cannot have both remove and include overlaps"
            )
            return -1
        config.include_overlaps = False
    if config.remove_backchannels:
        if config.include_backchannels:
            logging.getLogger(__name__).error(
                "Cannot have both remove and include backchannels"
            )
            return -1
        config.include_backchannels = False
    if config.include_end_bc_token:
        if "<ebc>" not in config.end_of_utterance_tokens:
            config.end_of_utterance_tokens.append("<ebc>")
    if config.include_yield_token:
        if "<yield>" not in config.end_of_utterance_tokens:
            config.end_of_utterance_tokens.append("<yield>")
    if config.include_overlap_token:
        if "<eint>" not in config.end_of_utterance_tokens:
            config.end_of_utterance_tokens.append("<eint>")

    if config.no_emp_tokens:
        if "<emp>" not in config.filter_special_tokens:
            config.filter_special_tokens.append("<emp>")
    if config.filter_bc_overlap_token:
        if "<eint>" not in config.filter_special_tokens:
            config.filter_special_tokens.append("<eint>")
        if "<ebc>" not in config.filter_special_tokens:
            config.filter_special_tokens.append("<ebc>")

    if config.datasets is not None and len(config.datasets) > 0:
        logging.getLogger(__name__).warning(
            "datasets is deprecated. Use train-datasets and eval-datasets. Defaulting to train-datasets and eval-datasets"
        )
        config.train_datasets = config.datasets
        config.eval_datasets = config.datasets

    return config


if __name__ == "__main__":
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module="transformers"
    )

    parser = build_parser()
    config = parser.parse_args()

    config = validate_args(config)
    logger.info(f"Config: {config}")

    if config == -1:
        logger.error(f"Invalid config setting {config}")
        exit(-1)

    if config.result_run != "":
        from pairwisegpt.config import DEFAULT, CONFIGS
        from common.utils import get_configs

        for run_config in get_configs(parser, CONFIGS, config, default=DEFAULT):
            if run_config.shutdown:
                os.system("shutdown 5")
                break

            main(run_config)
    else:
        logging.getLogger(__name__).info(f"{config}")
        main(config)
