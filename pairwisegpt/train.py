import random

import torch
import transformers
from torch.utils.data import DataLoader

import argparse
import logging
import json
import os
import warnings

from datetime import datetime

from data import PairwiseGenerationDM
from types import SimpleNamespace

from pairwisegpt import PairwiseGPT
# from pairwisegpt.evaluate import Analyser
from trainer import Trainer, get_abs_path

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def build_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Logger built")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bespoke Tart model used to predict turn-taking from linguistic features")
    parser.add_argument('--cuda', action="store_true",
                        help="true/false if cuda should be enabled")
    parser.add_argument('--load-model', action="store_true",
                        help="true/false if model should be loaded")
    parser.add_argument('--load-path', type=str, default='trained_model/',
                        help="load model config and weights from this file and ignore input configurations")
    parser.add_argument('--save-path', type=str, default='trained_model/',
                        help="model weights and config options save directory")

    parser.add_argument('--finetune', action="store_true",
                        help='true/false if BERT should be finetuned')
    parser.add_argument('--pretrained', type=str,
                        default='bert-base-uncased',
                        help="name of pretrained BERT model")
    parser.add_argument('--save-model-allowed', action='store_true')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0000625,
                        help="learning rate for non-cross-attention parameters in the model")
    parser.add_argument('--pretrained-learning-rate', type=float, default=0.0000625,
                        help="learning rate for cross-attention parameters in the model")
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--early-stop', type=int, default=5,
                        help='number of iterations without improvement for early stop')
    parser.add_argument('--weight-eos-token', type=float, default=1)
    parser.add_argument('--weight-reg-token', type=float, default=0.5)
    parser.add_argument('--weight-tokens', action='store_true')
    parser.add_argument('--remove-cross-attention', action='store_true')

    parser.add_argument('--evaluate', action='store_true',
                        help='model should only be evaluated. load-model and load-path should be set')

    parser.add_argument('--description', type=str, default='',
                        help="description of model")

    parser.add_argument('--run-name', type=str, default='',
                        help="set name of run")
    parser.add_argument('--result-run', type=str, default='',
                        help="set type of result run to go on")
    parser.add_argument('--detail', type=str, default='',
                        help="append to --run-name")

    # Wandb
    parser.add_argument('--log_interval', type=int, default=100,
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
    parser.add_argument('--yield-overlap-thresh', type=float, default=2,
                        help="number of seconds which overlap is from turn end to define turn end as yield")

    parser.add_argument('--include-speaker-embeddings', action="store_true",
                        help="add speaker tokens as token type ids")
    parser.add_argument('--individual-speaker-tokens', action="store_true",
                        help="use special speaker tokens for each speaker")

    parser.add_argument('--projection-labels', action='store_true',
                        help="add projection labels and convert to multitask learning")
    parser.add_argument('--remove-overlaps', action="store_true",
                        help="remove overlaps from data when parsing")
    parser.add_argument('--remove-backchannels', action="store_true",
                        help="remove backchannels from data when parsing")
    parser.add_argument('--remove-start-tokens', action="store_true",
                        help="remove start tokens")

    parser.add_argument('--serialise-data', action='store_true',
                        help="perform same perprocessing done for gptonly except pairwise")

    parser.add_argument('--no-emp-tokens', action='store_true')

    parser.add_argument('--include-overlap-token', action='store_true',
                        help="add special end token for completely overlapped dialog. Also updates metric generation")
    parser.add_argument('--include-yield-token', action='store_true',
                        help="add special end token for a turn end that is caused by an overlap or interruption. Also updates metric generation to include stats")
    parser.add_argument('--include-end-bc-token', action='store_true',
                        help="add special end token for a backchannel. Also updates metric generation to include stats")
    parser.add_argument('--filter-bc-overlap-token', action='store_true',
                        help="replaces <ebc> or <eint> with <emp>")
    parser.add_argument('--include-bc-token', action='store_true',
                        help="replace all backchannel words with specific <bc> token")

    parser.add_argument('--mask-attention-emp', action="store_true",
                        help="whether to mask attention for <emp>")
    parser.add_argument('--no-loss-emp', action="store_true",
                        help="whether to include loss for <emp> token")
    parser.add_argument('--remove-emp-metric-generation', action="store_true",
                        help="whether to include <emp> token as part of metric generation")
    parser.add_argument('--use-speaker-token-in-embedding',
                        action='store_true', help="using special speaker tokens")

    parser.add_argument('--rule-based-yield', action='store_true')

    parser.add_argument('--shutdown', action='store_true')
    parser.add_argument('--offline', action='store_true')

    parser.add_argument('--clone-self-cross', action='store_true',
                        help="experiment for replicating weights from self to cross attention")
    parser.add_argument('--evaluate-on-full', action='store_true')
    parser.add_argument('--evaluate-on-serialised', action="store_true")

    parser.add_argument('--random-seed', type=int, default=-1)

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


def analysis(config):
    logging.getLogger(__name__).info("model: Analysing model")

    if not config.load_model:
        logging.getLogger(__name__).error(f"Load model should be set")
        return

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
    model = PairwiseGPT(
        **vars(config)
    )

    model.to(config.device)

    test_ds = PairwiseGenerationDM(
        split="test",
        tokenizer=model.get_tokenizer(),
        overwrite=True if config.overwrite else False,
        max_length=config.max_length,
        keep_length=config.keep_length,
        overlap_length=config.overlap_length,
        datasets=config.datasets,
        basic_mode=config.serialise_data,
        **vars(config),
    )
    test_ds.prepare_data()
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        collate_fn=test_ds.collate_fn,
        num_workers=8,
        shuffle=True
    )

    analyser = Analyser(model, dataset=test_dl)
    analyser.turn_shift_projection()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def main(config):
    """
    Main entry function for a training or evaluation run.
    Reads `config` to determine the run type and parameters for the data/model/evaluation
    """
    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda) else "cpu"
    logging.getLogger(__name__).info(f"{config}")

    logging.getLogger(__name__).info(
        f"model: initialising model with name {config.run_name}")

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

    if config.evaluate and False:
        analysis(config)

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
    model = PairwiseGPT(
        **vars(config)
    )

    model.to(config.device)

    criterion = torch.nn.CrossEntropyLoss().to(config.device)

    new_params = []
    pretrained_params = []
    for name, param in model.named_parameters():
        if 'crossattention' in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    # Set different learning rates for different parameter groups
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': config.learning_rate, 'weight_decay': 1e-5},
        {'params': new_params, 'lr': config.pretrained_learning_rate, 'weight_decay': 1e-4}
    ])

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
            basic_mode=config.serialise_data,
            **vars(config),
        )
        train_ds.prepare_data()
        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            collate_fn=train_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )
        if config.evaluate_on_full and not config.evaluate_on_serialised:
            val_ds = PairwiseGenerationDM(
                split="val",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=False,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            val_ds.prepare_data()

            test_ds = PairwiseGenerationDM(
                split="test",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=False,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            test_ds.prepare_data()
        elif config.evaluate_on_serialised:
            val_ds = PairwiseGenerationDM(
                split="val",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=True,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            val_ds.prepare_data()

            test_ds = PairwiseGenerationDM(
                split="test",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=True,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            test_ds.prepare_data()
        else:
            val_ds = PairwiseGenerationDM(
                split="val",
                tokenizer=model.get_tokenizer(),
                basic_mode=config.serialise_data,
                **vars(config)
            )
            val_ds.prepare_data()

            test_ds = PairwiseGenerationDM(
                split="test",
                tokenizer=model.get_tokenizer(),
                basic_mode=config.serialise_data,
                **vars(config)
            )
            test_ds.prepare_data()

        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            collate_fn=val_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            collate_fn=test_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        scheduler = None

        logging.getLogger(__name__).info("model: train model")
        history = trainer.train(train_dl, val_dl, test_dl, scheduler=scheduler)
        wandb.finish()
    else:
        # Evaluate Model
        if config.evaluate_on_full:
            val_ds = PairwiseGenerationDM(
                split="val",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=False,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            val_ds.prepare_data()
            test_ds = PairwiseGenerationDM(
                split="test",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=False,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            test_ds.prepare_data()
        elif config.evaluate_on_serialised:
            val_ds = PairwiseGenerationDM(
                split="test",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=True,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            val_ds.prepare_data()
            test_ds = PairwiseGenerationDM(
                split="test",
                tokenizer=model.get_tokenizer(),
                overwrite=config.overwrite,
                basic_mode=True,
                include_end_bc_token=True,
                include_overlap_token=True,
                remove_start_tokens=True,
                filter_bc_overlap_token=config.filter_bc_overlap_token,
                max_length=config.max_length,
                keep_length=config.keep_length,
                overlap_length=config.overlap_length,
                datasets=config.datasets,
            )
            test_ds.prepare_data()

        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            collate_fn=val_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            collate_fn=test_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )
        logging.getLogger(__name__).info("model: evaluate model")
        if not config.load_model:
            logging.getLogger(__name__).error(
                "model: model is not being loaded")
            return None

        history = trainer.evaluate(val_dl, test_dl)
        wandb.finish()

    return history


def validate_args(config):
    return config


if __name__ == "__main__":
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module="transformers")

    build_logger()
    logger = logging.getLogger()

    parser = build_parser()
    config = parser.parse_args()

    logger.info(f"{config}")

    config = validate_args(config)
    if config == -1:
        logger.error(f"Invalid config setting {config}")
        exit(-1)

    if config.result_run != "":
        from config import get_configs

        for run_config in get_configs(parser, config):
            if run_config.shutdown:
                os.system('shutdown 5')
                break
            main(run_config)
    else:
        logging.getLogger(__name__).info(f"{config}")
        main(config)
