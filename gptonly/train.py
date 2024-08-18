import torch
from torch.utils.data import DataLoader

import argparse
import logging
import json
import os
import warnings
import random

from datetime import datetime

from data import PairwiseGenerationDM, GenerationDM
from types import SimpleNamespace

from gptonly import GPT
from trainer import Trainer, get_abs_path

from transformers import transformers
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


import wandb


def build_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Logger built")


def build_parser():
    parser = argparse.ArgumentParser(
        description="TurnGPT model used to predict turn-taking from linguistic features")
    parser.add_argument('--cuda', action="store_true",
                        help="true/false if cuda should be enabled")
    parser.add_argument('--load-model', action="store_true",
                        help="true/false if model should be loaded")
    parser.add_argument('--load-path', type=str, default='trained_model/',
                        help="load model config and weights from this file and ignore input configurations. --load-model should be set to parse the path")
    parser.add_argument('--save-path', type=str, default='trained_model/',
                        help="model weights and config options save directory")
    parser.add_argument('--save-model-allowed', action='store_true',
                        help='true/false to save model parameters')

    parser.add_argument('--finetune', action="store_true",
                        help='true/false if TurnGPT should be finetuned')
    parser.add_argument('--pretrained', type=str,
                        default='gpt2',
                        help="name of pretrained TurnGPT model")

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=6.75e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--early-stop', type=int, default=2,
                        help='number of iterations without improvement for early stop')
    parser.add_argument('--weight-eos-token', type=float, default=1,
                        help="weight for the <eot> or other special end-of-turn tokens")
    parser.add_argument('--weight-reg-token', type=float,
                        default=0.5, help="all lexical content tokens")
    parser.add_argument('--weight-tokens', action='store_true',
                        help="true/false to weight specific tokens")

    parser.add_argument('--evaluate', action='store_true',
                        help='model should only be evaluated. load-model and load-path should be set')

    parser.add_argument('--description', type=str, default='',
                        help="description of model")
    parser.add_argument('--run-name', type=str,
                        default='run-name is input into wandb')
    parser.add_argument('--result-run', type=str, default='',
                        help='what type of result run to run based on config.py')

    # Wandb
    parser.add_argument('--log-interval', type=int, default=100,
                        help="frequency with which to report logs to wandb")

    # Dataset
    parser.add_argument('--overwrite', action='store_true',
                        help="overwrite and regenerate dataset")
    parser.add_argument('--dev-mode', action='store_true',
                        help="decrease dataset size to test post-processing steps/disable writing to wandb")
    parser.add_argument('--datasets', nargs="+", help="Datasets to use",
                        default=["switchboard"])

    parser.add_argument('--max-length', type=int, default=256,
                        help="max length of a sequence")
    parser.add_argument('--keep-length', type=int, default=64,
                        help="minimum length of a sequence")
    parser.add_argument('--overlap-length', type=int, default=10,
                        help="number of tokens to overlap between sequences")

    parser.add_argument('--include-speaker-embeddings', action="store_true",
                        help="add speaker tokens as token type ids")
    parser.add_argument('--individual-speaker-tokens', action="store_true",
                        help="true/false use a special token for each speaker (not 1 or 2)")

    parser.add_argument('--remove-backchannels', action="store_true",
                        help="remove backchannels for switchboard")
    parser.add_argument('--remove-overlaps', action="store_true",
                        help="remove overlaps for switchboard")

    parser.add_argument('--include-yield-tokens', action="store_true",
                        help="add end yield tokens")
    parser.add_argument('--use-speaker-token-in-embedding',
                        action='store_true', help="uses special speaker tokens in embedding rather than a generic ie. <speaker1> <speaker2>")

    # For baseline perplexity measurements without new tokens
    parser.add_argument('--remove-special-tokens', action='store_true',
                        help="removes special tokens from dataset and tokenizer")

    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--random-seed', type=int, default=-1)

    """
    Ignore below; experimental work
    """
    parser.add_argument('--weight-projection', type=float, default=0.5)
    parser.add_argument('--projection-labels', action='store_true',
                        help="add projection labels and convert to multitask learning")
    parser.add_argument('--normalize-time', action="store_true",
                        help="normalize projection labels")
    parser.add_argument('--categorize-projection', action="store_true",
                        help="categorize the time blocks")
    parser.add_argument('--category-bins', type=int, default=5,
                        help="decide number of bins required for time until ts")

    return parser


def get_latest_model(path):
    list_dir = os.listdir(path)
    latest_model = None
    latest_index = -1
    for item in list_dir:
        if item[:5] == 'model':
            index = int(''.join(x for x in item if x.isdigit()))
            if latest_model is None or index > latest_index:
                latest_index = index
                latest_model = item

    if latest_model is None:
        raise RuntimeError("model file not found")
    return os.path.join(path, latest_model)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def main(config):
    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda) else "cpu"

    logging.getLogger(__name__).info(f"{config}")
    logging.getLogger(__name__).info("model: initialising model")

    if config.run_name == "":
        name = input("Enter new change(s) for wandb run: ")
        if name == "":
            name = None
        config.run_name = name

    wandb.init(
        config=config,
        entity="leishy333",
        name=config.run_name,
        mode="online" if not config.offline else "offline"
    )
    logging.getLogger(__name__).info(f"model: set run name {config.run_name}")

    if config.random_seed != -1:
        logging.getLogger(__name__).info(
            f"model set random seed {config.random_seed}")
        set_seed(config.random_seed)

    if config.load_model:
        load_path = get_abs_path(config.load_path)
        logging.getLogger(__name__).info(
            f"model: loading model from {load_path}")

        with open(os.path.join(load_path, "config.json")) as f:
            new_config = SimpleNamespace(**json.load(f))
            for arg in vars(new_config):
                config.arg = getattr(new_config, arg)

        logging.getLogger(__name__).info(f"Loaded config: {config}")

    logging.getLogger(__name__).info(
        f"Loaded model: gpt with finetuning: {config.finetune}")
    model = GPT(
        **vars(config)
    )

    model.to(config.device)

    # criterion = torch.nn.BCEWithLogitsLoss(
    #    pos_weight=torch.FloatTensor([config.loss_weight]).to(config.device))
    criterion = torch.nn.CrossEntropyLoss().to(config.device)
    # , weight_decay=config.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if config.load_model:
        trainer = Trainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          config=config,
                          load_from_checkpoint=get_latest_model(load_path),
                          **vars(config)
                          )
    else:
        trainer = Trainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          config=config,
                          **vars(config)
                          )

    if not config.evaluate:
        train_ds = PairwiseGenerationDM(
            split="train",
            tokenizer=model.get_tokenizer(),
            load_metrics_from_all=True,
            basic_mode=True,
            combine_speaker=True,
            **vars(config)
        )
        train_ds.prepare_data()
        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            collate_fn=train_ds.collate_fn,
            num_workers=16,
            shuffle=False
        )
        val_ds = PairwiseGenerationDM(
            split="val",
            tokenizer=model.get_tokenizer(),
            load_metrics_from_all=True,
            basic_mode=True,
            combine_speaker=True,
            **vars(config)
        )
        val_ds.prepare_data()
        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            collate_fn=val_ds.collate_fn,
            num_workers=16,
            shuffle=False
        )
        test_ds = PairwiseGenerationDM(
            split="test",
            tokenizer=model.get_tokenizer(),
            load_metrics_from_all=True,
            basic_mode=True,
            combine_speaker=True,
            **vars(config)
        )
        test_ds.prepare_data()
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            collate_fn=test_ds.collate_fn,
            num_workers=16,
            shuffle=False
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        scheduler = None

        logging.getLogger(__name__).info("model: train model")
        history = trainer.train(train_dl, val_dl, test_dl, scheduler=scheduler)
        wandb.finish()
    else:
        test_ds = PairwiseGenerationDM(
            split="test",
            tokenizer=model.get_tokenizer(),
            load_metrics_from_all=True,
            basic_mode=True,
            combine_speaker=True,
            **vars(config)
        )
        test_ds.prepare_data()
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            collate_fn=test_ds.collate_fn,
            num_workers=8,
            shuffle=True
        )

        logging.getLogger(__name__).info("model: evaluate model")
        if not config.load_model:
            logging.getLogger(__name__).error(
                "model: model is not being loaded")
            return None

        history = trainer.evaluate(test_dl)

    return history


if __name__ == "__main__":
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module="transformers")

    build_logger()

    parser = build_parser()
    config = parser.parse_args()

    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda) else "cpu"

    if config.result_run != "":
        from config import get_configs

        for run_config in get_configs(parser, config):
            logging.getLogger(__name__).info(f"{run_config}")
            main(run_config)
    else:
        logging.getLogger(__name__).info(f"{config}")
        main(config)
