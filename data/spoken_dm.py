import torch
import random
import logging
import os
import argparse
import torch.multiprocessing as mp
import itertools
import uuid
import threading
import pickle
import json

from switchboard import SwitchboardDataset
from fisher import FisherDataset
from edacc import EdAccDataset
from data.utils import get_abs_path, pp_pair_dialogs
from aligned_process import AlignedProcess
from serialised_process import SerialisedProcess, to_serialised_process_type
from base import DialogDMInterface, Datasets, TurnID, Turn
from utils import get_logger
from pairwisegpt.tokenizer import SpokenDialogTokenizer
from huggingface_hub import snapshot_download

from datasets import (
    Dataset as HFDataset,
    concatenate_datasets as hf_concatenate_datasets,
)

from torch.utils.data import DataLoader, Dataset
from base import ConcatDatasetLoader as ConcatDataset


def concatenate_datasets(datasets):
    if type(datasets) != list:
        raise ValueError("Datasets must be a list")
    if len(datasets) == 0:
        raise ValueError("Datasets must not be empty")

    dataset = datasets[0]
    if len(datasets) == 1:
        return dataset

    is_hf = all([isinstance(ds, HFDataset) for ds in datasets[1:]])
    is_torch = all([isinstance(ds, Dataset) for ds in datasets[1:]])

    if not (is_hf or is_torch):
        raise ValueError(
            "Datasets must be of the same type", [type(ds) for ds in datasets[1:]]
        )

    if is_torch:
        datasets = datasets[0].extend(datasets[1:])
    elif is_hf:
        dataset = hf_concatenate_datasets(datasets)

    return dataset


class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *_, **kwargs):
        return self.generator(**kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")


CACHE_PATH = get_abs_path(".cache")
logger = get_logger(__name__, level=logging.DEBUG)


def index(obj, comp):
    return obj.index(comp)


def pool_init(lock):
    global curr_conv_id_lock
    curr_conv_id_lock = lock


def BuildProcess(
    tokenizer,
    method="serialised",
    serialised=None,
    combine_speaker=False,
    include_partial_overlaps=False,
    filter_special_tokens=[],
    split_utt=True,
    *args,
    **kwargs,
):
    if isinstance(serialised, bool):
        serialised = None

    if method == "serialised":
        if serialised is None:
            logger.warning(f"Setting serialised to {serialised}")
            serialised = "turn-end"

        return SerialisedProcess(
            tokenizer,
            combine_speaker=combine_speaker,
            no_overlap=False,  # Ensures that lexical content and <eot> token can be aligned
            end_on=to_serialised_process_type(serialised),
            split_utt=split_utt,
            filter_special_tokens=filter_special_tokens,
            *args,
            **kwargs,
        )
    elif method == "aligned":
        if not include_partial_overlaps:
            logger.warning("Setting include_partial_overlaps to True")
            include_partial_overlaps = True

        return AlignedProcess(
            tokenizer,
            split_utt=split_utt,
            include_partial_overlaps=include_partial_overlaps,
            filter_special_tokens=filter_special_tokens,
            *args,
            **kwargs,
        )

    raise ValueError(f"No process specified: {args} {kwargs}")


def build_path(idx, max_num=100000):
    string_idx = str(idx)
    string_idx = string_idx.rjust(len(str(max_num)), "0")
    return os.path.join(string_idx[: len(string_idx) // 2 + 1], f"{string_idx}")


def chunkify(gen, length=-1, N=-1):
    iterable = iter(gen)
    if N > 0 and length == -1:
        l = len(gen) // N

    while True:
        try:
            if N > 0 and length == -1:
                yield itertools.chain(
                    (next(iterable),), itertools.islice(iterable, l - 1)
                )
            elif length > 0 and N == -1:
                yield itertools.chain(
                    (next(iterable),), itertools.islice(iterable, length - 1)
                )
            else:
                raise ValueError("Either length or N must be provided")
        except StopIteration:
            return


class SpokenDM(DialogDMInterface):
    def __init__(
        self,
        split="train",
        tokenizer=None,
        device="cuda:0",
        save_data=True,
        load_from_hub=False,
        load_from_cache=False,
        max_length=256,
        keep_length=64,
        overlap_length=10,
        filter_special_tokens=[],
        split_utt=True,
        keep_in_memory=False,
        dev_mode=False,
        parse_dialogs=None,
        datasets=[Datasets.SWITCHBOARD],
        include_bc_token=False,
        end_of_utterance_tokens=["<eint>", "<ebc>"],  # Can include <yield>
        *args,
        **kwargs,
    ):
        self.tokenizer: SpokenDialogTokenizer = tokenizer
        if self.tokenizer is None:
            logger.warning("No tokenizer provided, using default")
            tokens = end_of_utterance_tokens + ["<eot>", "<emp>"]
            if include_bc_token:
                tokens.append("<bc>")
            self.tokenizer = SpokenDialogTokenizer(tokens=tokens)
        self.device = device

        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Split {split} not supported")
        self.split = split

        self.datasets = {}
        self.data = []
        self.data_length = 0

        self.save_path = CACHE_PATH
        if self.save_path == CACHE_PATH:
            tokenizer_string = self.tokenizer.__str__()
            if "(" in tokenizer_string:
                dirname = tokenizer_string[: index(tokenizer_string, "(")]
            else:
                dirname = (
                    "GPT2TokenizerFast"
                    if "gpt" in self.tokenizer.__str__().lower()
                    else "LlamaTokenizerFast"
                )
            self.save_path = os.path.join(CACHE_PATH, "conversational", dirname)
            logger.warning(
                f"No save path provided. Using default save path: {self.save_path}"
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.save_data = save_data
        self.load_from_cache = load_from_cache
        self.keep_in_memory = keep_in_memory

        self.load_from_hub = load_from_hub

        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length

        self.dev_mode = dev_mode
        self.parse_dialogs = parse_dialogs

        self.filter_special_tokens = filter_special_tokens

        self.processor = BuildProcess(
            self.tokenizer,
            max_length=self.max_length,
            keep_length=self.keep_length,
            overlap_length=self.overlap_length,
            split_utt=split_utt,
            filter_special_tokens=self.filter_special_tokens,
            include_bc_token=include_bc_token,
            *args,
            **kwargs,
        )

        self.dataset_keys = datasets
        self.dataset = Dataset()
        self.ts = {}
        self.init_datasets()

        self.log_info = {
            "ebc_count": 0,
            "eint_count": 0,
            "eot_count": 0,
            "yield_count": 0,
            "turn_count": 0,
        }

        self.key_set = {
            "input_ids",
            "attention_mask",
            "speaker_ids",
            "token_type_ids",
            "other_token_type_ids",
            "turn_ids",
            "conv_id",
        }
        self.warned_key_set = {"time_until_ts", "conv_id"}

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        if self.data is None or len(self.data) == 0:
            logger.info(
                f"Loading data from {self.get_save_load_dir()} with initial __getitem__ call"
            )
            self.setup()

        if len(self.data) <= index:
            raise IndexError(
                f"Index {index} out of bounds with length {len(self.data)}"
            )

        return self.data[index]

    def __str__(self):
        datasets = "".join(ds.name for ds in self.dataset_keys)
        include_yield_token = self.tokenizer.convert_tokens_to_ids("<yield>") not in {
            self.tokenizer.pad_token,
            self.tokenizer.unk_token,
            self.tokenizer.eos_token,
        }
        debug = f"debug={int(self.dev_mode)}"
        include_yield_token = f"include_yield_token={int(include_yield_token)}"
        return f"GenerationDM-{self.split}-{datasets}-{debug}-{include_yield_token}-{self.processor.__hash__()}"

    def get_turn_by_turn_id(self, turn_id: int | TurnID) -> Turn:
        if isinstance(turn_id, TurnID):
            turn_id = turn_id.value

        if turn_id not in self.ts:
            raise ValueError(f"Turn ID {turn_id} not found in dataset")

        return self.ts[turn_id]

    def reset(self):
        logger.info("Resetting SpokenDM")
        self.data = []
        self.ts = {}
        self.datasets = {}
        self.data_length = 0

    def init_datasets(self):
        logger.info("Loading SpokenDM datasets...")
        for ds in self.dataset_keys:
            if ds == Datasets.SWITCHBOARD:
                self.datasets[ds] = SwitchboardDataset(
                    split=self.split,
                    pairwise=True,
                    dev_mode=self.dev_mode,
                    parse_dialogs=self.parse_dialogs,
                    keep_in_memory=False,
                )
            elif ds == Datasets.FISHER:
                self.datasets[ds] = FisherDataset(
                    split=self.split,
                    dev_mode=self.dev_mode,
                    pairwise=True,
                    keep_in_memory=False,
                )
            elif ds == Datasets.EDACC:
                self.datasets[ds] = EdAccDataset(
                    split=self.split,
                    dev_mode=self.dev_mode,
                )
            else:
                raise ValueError(f"Dataset {ds} not supported")

    def update_log_info(self, d):
        ebc_id, eint_id, eot_id, yield_id, _ = self.tokenizer.convert_tokens_to_ids(
            ["<ebc>", "<eint>", "<eot>", "<yield>", "<emp>"]
        )

        for speaker in d.keys():
            self.log_info["eot_count"] += torch.sum(
                d[speaker]["input_ids"] == eot_id
            ).item()
            if self.tokenizer.is_special_token("<ebc>"):
                self.log_info["ebc_count"] += torch.sum(
                    d[speaker]["input_ids"] == ebc_id
                ).item()
            if self.tokenizer.is_special_token("<eint>"):
                self.log_info["eint_count"] += torch.sum(
                    d[speaker]["input_ids"] == eint_id
                ).item()
            if self.tokenizer.is_special_token("<yield>"):
                self.log_info["yield_count"] += torch.sum(
                    d[speaker]["input_ids"] == yield_id
                ).item()

    def output_log_info(self):
        logger.info(f"Split {self.split} has the following stats:")
        for k, v in self.log_info.items():
            logger.info(f"Split {self.split} {k}: {v}")

    def collate_fn(self, batch):
        def collate_fn_channel(batch):
            ret = self.tokenizer.pad(
                {"input_ids": [b["input_ids"][: self.max_length] for b in batch]},
                padding="max_length",
                max_length=self.max_length,
            )

            for key in self.key_set:
                if key not in ret:
                    if key not in self.warned_key_set:
                        self.warned_key_set.add(key)
                        logger.warning(
                            f"Key {key} not found in batch, setting to torch.zeros_like() with shape {ret['input_ids'].shape}"
                        )

                    if key == "conv_id":
                        ret[key] = "placeholder"
                        continue

                    ret[key] = torch.zeros_like(ret["input_ids"])

            for key, value in batch[0].items():
                if key == "input_ids" or key == "attention_mask":
                    continue
                if key == "conv_id":
                    first_conv_id = batch[0][key]
                    ret[key] = [b[key] if key in b else first_conv_id for b in batch]
                    continue
                if not isinstance(value, torch.Tensor) or key not in self.key_set:
                    continue

                ret[key] = self.tokenizer.pad(
                    {"input_ids": [b[key][: self.max_length] for b in batch]},
                    padding="max_length",
                    max_length=self.max_length,
                )["input_ids"]

            for k, v in ret.items():
                if isinstance(v, torch.Tensor):
                    ret[k] = v.clone().detach()
                else:
                    ret[k] = v

            return ret

        result = {}
        for key in batch[0].keys():
            if "speaker" in key:
                result[key] = collate_fn_channel([x[key] for x in batch])

        return result

    def get_save_load_dir(self, dir="", with_self=True):
        save_load_dir = get_abs_path(self.save_path)
        if dir != "":
            save_load_dir = get_abs_path(dir)

        if not os.path.exists(save_load_dir):
            os.mkdir(save_load_dir)

        if with_self:
            save_load_dir = os.path.join(save_load_dir, self.__str__())
            if not os.path.exists(save_load_dir):
                os.mkdir(save_load_dir)
            return save_load_dir

        return save_load_dir

    def get_save_load_path(self, dir="", file_type="", ext=""):
        save_load_dir = self.get_save_load_dir(dir=dir, with_self=False)

        ext = "." + ext if len(ext) > 0 and ext[0] != "." else ext
        file_type = f"-{file_type}" if len(file_type) > 0 else ""
        return os.path.join(save_load_dir, f"{str(self)}{file_type}{ext}")

    def prepare_data(self):
        if self.load_from_cache:
            logger.info(f"Loading data from cache at file {self.get_save_load_path()}")
            if self.setup():
                logger.info("Data loaded from cache")
                self.output_log_info()
                return

            logger.info("Loading data from cache failed so generate data")
        elif self.load_from_hub:
            logger.info("Loading data from HuggingFace Datasets")
            repo_local_path = snapshot_download(
                repo_id="seanleishman/PairwiseTurnGPT",
                repo_type="dataset",
                token=os.getenv("HUGGINGFACE_TEST"),
            )
            if not os.path.exists(repo_local_path):
                raise FileNotFoundError(f"Path {repo_local_path} not found")

            folder_name = os.path.join(
                repo_local_path, self.get_save_load_dir(), "final.hf"
            )
            turns_folder_name = os.path.join(
                repo_local_path, self.get_save_load_dir(), "turns.hf"
            )

            if os.path.exists(folder_name):
                self.data = HFDataset.load_from_disk(folder_name)
                self.data_length = len(self.data)

                if os.path.exists(turns_folder_name):
                    self.ts = HFDataset.load_from_disk(turns_folder_name)
                else:
                    self.ts = self._generate_turns(self.data)

                return

            logger.info(
                "Loading data from HuggingFace Datasets failed so generate data"
            )

        self._prepare_data()

    def _prepare_data(self):
        # Remove any existing Dataset
        dataset_path = self.get_save_load_dir()
        logger.info(f"Removing existing dataset at {dataset_path}")
        if os.path.exists(dataset_path):
            for dirpath, _, filenames in os.walk(dataset_path, topdown=False):
                for filename in filenames:
                    logger.info(f"Removing file: {os.path.join(dirpath, filename)}")
                    os.remove(os.path.join(dirpath, filename))
                logger.info(f"Removing directory: {dirpath} from paths {dirpath}")
                os.rmdir(dirpath)

        self.init_datasets()
        for ds in self.datasets.values():
            ds()

        logger.info("Completed loading datasets")

        self.dataset = ConcatDataset(self.datasets.values())
        self._process_data()
        self.output_log_info()
        if not self.keep_in_memory:
            self.data = []
            self.ts = {}
            self.dataset = Dataset()

        logger.info("Data preparation complete")

    def _generate_turns(self, data):
        NUM_WORKERS = min(24, os.cpu_count() - 1) if not self.dev_mode else 1

        def _gen_ts(data):
            for d in data:
                for turn in d["turns"]:
                    yield turn

        logger.info("Creating Turn objects from data")
        ts = HFDataset.from_generator(
            _gen_ts,
            gen_kwargs={"data": self.data},
            num_proc=NUM_WORKERS,
            keep_in_memory=True,
        )
        ts.save_to_disk(
            os.path.join(self.get_save_load_dir(), "turns.hf"), num_shards=NUM_WORKERS
        )
        logger.info(f"Turns saved to {self.get_save_load_dir()}")

        logger.info(f"Saving turn-shift mapping to {self.get_save_load_dir()}")
        mapping = {i: t["id"] for i, t in enumerate(ts)}

        def _save_mapping(ts):
            mapping = {i: t["id"] for i, t in enumerate(ts)}
            pickle.dump(
                mapping,
                open(os.path.join(self.get_save_load_dir(), "mapping.pkl"), "wb"),
            )

        threading.Thread(target=_save_mapping, args=(ts,)).start()

        self.ts_mapping = mapping
        self.ts = ts

    def _process_data(self):
        logger.info("Processing data...")

        NUM_WORKERS = min(24, os.cpu_count() - 1) if not self.dev_mode else 1
        STEP = len(self.dataset) // NUM_WORKERS

        starts = [i * STEP for i in range(0, NUM_WORKERS)]
        steps = [STEP for _ in range(len(starts))]

        logger.info(
            f"Processing {len(starts)} chunks with {NUM_WORKERS} workers from {len(self.dataset)} and step {STEP}"
        )

        ds = HFDataset.from_generator(
            self.processor.process,
            gen_kwargs={
                "datasets": [
                    (self.dataset, starts[i], steps[i]) for i in range(len(starts))
                ],
                "_misc": random.random(),  # [random.random() for _ in range(len(starts))],
            },
            num_proc=NUM_WORKERS,
            keep_in_memory=True,
        )
        ds.save_to_disk(
            os.path.join(self.get_save_load_dir(), "final.hf"), num_shards=NUM_WORKERS
        )

        self._save_config(os.path.join(self.get_save_load_dir(), "config.json"))

        self.data = ds
        self.data_length = len(ds)

        self._generate_turns(self.data)

    def _save_config(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "max_length": self.max_length,
                    "keep_length": self.keep_length,
                    "overlap_length": self.overlap_length,
                    "filter_special_tokens": self.filter_special_tokens,
                    "processor": self.processor.config_to_dict(),
                    "datasets": [ds.name for ds in self.dataset_keys],
                },
                f,
                indent=4,
            )

    def setup(self) -> bool:
        filename = os.path.join(self.get_save_load_dir(), "final.hf")
        try:
            logger.info(f"Loading data from {filename} ...")
            self.data = HFDataset.load_from_disk(filename).with_format("torch")
            self.data_length = len(self.data)

            try:
                self.ts = HFDataset.load_from_disk(
                    os.path.join(self.get_save_load_dir(), "turns.hf")
                )
                self.ts_mapping = pickle.load(
                    open(os.path.join(self.get_save_load_dir(), "mapping.pkl"), "rb")
                )
            except FileNotFoundError:
                self._generate_turns(self.data)

            return True
        except FileNotFoundError as e:
            logger.error(f"File not found: {filename} and e={e}")

        return False

    def show_input(self, batch=None, conv_id=None, **kwargs):
        for item in self.show_input_iterator(batch=batch, conv_id=conv_id, **kwargs):
            if item is None:
                break

            continue

    def show_input_iterator(self, batch=None, conv_id=None, allow_multi_lines=True):
        if batch is None and conv_id is None:
            raise ValueError("Either batch or conv_id must be provided")

        if batch is None:
            batch = [x for x in self.data if conv_id in x["speakerA"]["conv_id"]]
            if len(batch) == 0:
                logger.error(f"Conversation ID {conv_id} not found")
                return None
            elif len(batch) > 1:
                logger.warning(
                    f"Multiple conversations found with ID {conv_id}. Picking first conversation with id {batch[0]['speakerA']['conv_id']}"
                )
            batch = batch[0]
            logger.debug(
                f"Conversation ID {conv_id} found in batch with ID {batch['speakerA']['conv_id']}"
            )

        input_idsA = batch["speakerA"]["input_ids"]
        input_idsB = (
            batch["speakerB"]["input_ids"]
            if "speakerB" in batch
            else torch.zeros_like(input_idsA)
        )

        timingsA = batch["speakerA"]["timings"]
        timingsB = (
            batch["speakerB"]["timings"]
            if "speakerB" in batch
            else [(-1, -1) for _ in range(len(timingsA))]
        )
        typesA = batch["speakerA"]["token_type_ids"]
        typesB = (
            batch["speakerB"]["token_type_ids"]
            if "speakerB" in batch
            else torch.zeros_like(typesA)
        )

        otherA = batch["speakerA"]["other_token_type_ids"]
        otherB = (
            batch["speakerB"]["other_token_type_ids"]
            if "speakerB" in batch
            else torch.zeros_like(otherA)
        )

        turn_idsA = batch["speakerA"]["turn_ids"]
        turn_idsB = (
            batch["speakerB"]["turn_ids"]
            if "speakerB" in batch
            else torch.zeros_like(turn_idsA)
        )

        speaker_idsA = batch["speakerA"]["speaker_ids"]
        speaker_idsB = (
            batch["speakerB"]["speaker_ids"]
            if "speakerB" in batch
            else torch.zeros_like(speaker_idsA)
        )

        start = 0
        end = start + len(input_idsA)

        print(
            "Conversation ID: ",
            batch["speakerA"]["conv_id"],
            " with timing from ",
            timingsA[:2],
            " to ",
            timingsA[-2:],
        )

        while True:
            othersA = {}
            othersB = {}
            if "interrupt_points" in batch["speakerA"]:
                othersA = {
                    "int_point": batch["speakerA"]["interrupt_points"],
                }
                int_pointsB = (
                    batch["speakerB"]["interrupt_points"]
                    if "speakerB" in batch
                    else torch.zeros_like(input_idsA)
                )
                othersB = {
                    "int_point": int_pointsB,
                }

            othersA["turn_ids"] = turn_idsA
            othersA["turn_end_types"] = otherA
            othersA["speaker_ids"] = speaker_idsA

            othersB["turn_ids"] = turn_idsB
            othersB["turn_end_types"] = otherB
            othersB["speaker_ids"] = speaker_idsB

            _, columns, _ = pp_pair_dialogs(
                self.tokenizer,
                input_idsA,
                timings=timingsA,
                start=start,
                token_types=typesA,
                speaker="A",
                others=othersA,
                width=os.get_terminal_size().columns if allow_multi_lines else -1,
            )
            start, _, _ = pp_pair_dialogs(
                self.tokenizer,
                input_idsB,
                timings=timingsB,
                start=start,
                token_types=typesB,
                others=othersB,
                speaker="B",
                columns=columns,
                width=os.get_terminal_size().columns if allow_multi_lines else -1,
            )
            print()

            if start >= end:
                break

            yield start

        print("------------------------------------")


if __name__ == "__main__":
    to_enum = {
        "switchboard": Datasets.SWITCHBOARD,
        "fisher": Datasets.FISHER,
        "edacc": Datasets.EDACC,
    }

    # Set here to avoid conflicting with requirements of DataLoader (hangs otherwise)
    # Might not work when load_from_cache is set to False
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )  # avoid hitting the open file limit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--process",
        choices=["serialised", "aligned", "future"],
        default=None,
        help="The process used to generate data. `serialised' processes the data in a serialised manner for use with turngpt. `aligned' processes the data in an aligned manner for use with pairwisegpt. `future' processes the data with future context for futture work",
    )

    future_group = parser.add_argument_group("future")
    future_group.add_argument(
        "--summarize-method",
        choices=["NONE", "HF_PROMPT"],
        default="none",
        help="The summarization method used to generate data for `future'. `NONE' does not use any summarization. `HF_PROMPT' uses the HuggingFace prompt-based summarization method",
    )
    serialised_group = parser.add_argument_group("serialised")
    serialised_group.add_argument(
        "--serialised",
        choices=["turn-end", "turn-start"],
        default=None,
        help="turn-end is the default which pushes the start of a turn to the end of the prior turn. `turn-start' (depreacted) pushes the start of a turn to the start of the prior turn.",
    )
    serialised_group.add_argument(
        "--combine-speaker",
        action="store_true",
        help="Combine speaker A and B channels into a single channel",
    )
    parser.add_argument(
        "--show-output", action="store_true", help="Show the processing output"
    )
    parser.add_argument(
        "--write-output",
        help="Write the processing output. Either requires `multi` or `single`",
        choices=["multi", "single"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["switchboard", "fisher", "edacc"],
        default=["switchboard"],
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test"
    )

    split_utt_group = parser.add_argument_group("split-utt")
    split_utt_group.add_argument(
        "--split-utt",
        action="store_true",
        help="Split utterances into separate turns. Required for use of `max-length', `keep-length', and `overlap-length' parameters",
    )
    split_utt_group.add_argument("--max-length", type=int, default=256)
    split_utt_group.add_argument("--keep-length", type=int, default=64)
    split_utt_group.add_argument("--overlap-length", type=int, default=10)

    aligned_group = parser.add_argument_group("aligned")
    aligned_group.add_argument(
        "--include-partial-overlaps",
        action="store_true",
        help="Required for aligned process",
    )
    aligned_group.add_argument(
        "--include-backchannels",
        action="store_true",
        help="Adds backchannels back to the aligned process",
    )
    aligned_group.add_argument(
        "--include-overlaps",
        action="store_true",
        help="Adds overlaps back to the aligned process",
    )

    aligned_group.add_argument(
        "--yield-int-thresh",
        type=float,
        default=0.2,
        help="Threshold of the overlap between the beginging of a turn for labelling the other turn-end as a yield",
    )

    aligned_group.add_argument(
        "--yield-overlap-thresh",
        type=float,
        default=2,
        help="Threshold of a complete overlap to a turn-end for labelling that turn-end as a yield",
    )

    aligned_group.add_argument(
        "--replace-bc-token",
        action="store_true",
        help="Replace backchannel turns with the <bc> token",
    )

    parser.add_argument(
        "--load-from-cache",
        action="store_true",
        help="Load data from cache",
    )

    aligned_group.add_argument(
        "--include-yield-token",
        action="store_true",
        help="Include the <yield> token in the tokenizer",
    )

    parser.add_argument(
        "--dev-mode", action="store_true", help="Run in development mode with less data"
    )
    args = parser.parse_args()

    gd = SpokenDM(
        serialised=args.serialised,
        datasets=[to_enum[ds] for ds in args.datasets],
        include_partial_overlaps=args.include_partial_overlaps,
        include_backchannels=args.include_backchannels,
        include_overlaps=args.include_overlaps,
        yield_int_thresh=args.yield_int_thresh,
        yield_overlap_thresh=args.yield_overlap_thresh,
        include_bc_token=args.replace_bc_token,
        dev_mode=args.dev_mode,
        load_from_hub=False,
        split=args.split,
        load_from_cache=args.load_from_cache,
        split_utt=args.split_utt,
        combine_speaker=args.combine_speaker,
        max_length=args.max_length,
        keep_length=args.keep_length,
        overlap_length=args.overlap_length,
        method=args.process,
        summarization_method=args.summarize_method,
        end_of_utterance_tokens=(
            ["<ebc>", "<eint>", "<yield>"]
            if args.include_yield_token
            else ["<ebc>", "<eint>"]
        ),
    )
    gd.prepare_data()

    if args.show_output:
        i = 0
        logger.info(
            f"Available conversations: {[g["speakerA"]["conv_id"] for g in gd][:10]} ...",
        )
        while True:
            input_string = input(
                "Press enter to continue or q to quit or c to enter conv_id: "
            )

            if input_string == "c":
                conv_id = input("Enter conversation ID: ")
                gd.show_input(conv_id=conv_id)
            elif input_string == "q":
                break
            else:
                gd.show_input(gd[i])
                i += 1
    elif args.write_output is not None:
        for i in range(len(gd)):
            gd.show_input(gd[i], allow_multi_lines=args.write_output == "multi")

    else:
        dl = DataLoader(gd, batch_size=4, collate_fn=gd.collate_fn, shuffle=True)
