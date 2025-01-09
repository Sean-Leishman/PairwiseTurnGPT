import torch
import logging
import os
import re

from torch.utils.data import ConcatDataset
from typing import List, Dict

from pairwisegpt.tokenizer import SpokenDialogTokenizer

from base import DialogDMInterface, Datasets
from utils import get_abs_path
from conversational.curiosity_dialog import CuriosityDialog
from conversational.daily_dialog import DailyDialog
from conversational.multiwoz_v2 import MultiWozV2
from conversational.metawoz import MetaWoz
from conversational.taskmaster import Taskmaster1
from conversational.taskmaster import Taskmaster2
from conversational.taskmaster import Taskmaster3

CACHE_PATH = get_abs_path(".cache")


class ConversationalProcessor:
    def __init__(
        self, tokenizer, max_length, keep_length, overlap_length, combine_speaker
    ):
        self.logger = logging.getLogger(__name__)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length

        self.combine_speaker = combine_speaker

        self.emp_token_id = self.tokenizer.convert_tokens_to_ids("<emp>")

        self.logger.info(
            f"Initialise ConversationalProcessor with tokenizer; max_length {self.max_length}; keep_length {self.keep_length}; overlap_length {self.overlap_length}; combine_speaker {self.combine_speaker}"
        )

    def process(self, dataset):
        data = []
        for dialog_obj in dataset:
            dialog = dialog_obj["dialog"]
            dialog_filtered = self.filter_empty_turns(dialog)
            data.extend(self.process_dialog(dialog_filtered))

        return data

    def process_dialog(self, dialog) -> List[Dict[str, torch.Tensor]]:
        data = []
        t = self.tokenizer(dialog)

        for i in range(0, len(t["input_ids"]), self.max_length - self.overlap_length):
            end_idx = max(i + self.max_length, len(t))
            if end_idx - i < self.keep_length:
                break

            sample = {
                "input_ids": torch.tensor(
                    t["input_ids"][i:end_idx], device=self.tokenizer.device
                ),
                "speaker_ids": torch.tensor(
                    t["speaker_ids"][i:end_idx], device=self.tokenizer.device
                ),
                "conv_id": t["conv_id"] if "conv_id" in t else "NULL",
            }

            if self.combine_speaker:
                data.append({"speakerA": sample})
            else:
                data.append(self.separate_by_speaker(sample))

        return data

    def separate_by_speaker(self, sample):
        input_ids = sample["input_ids"]
        speaker_ids = sample["speaker_ids"]

        speakerA = torch.eq(speaker_ids, self.tokenizer.sp1_token_id)
        speakerB = torch.eq(speaker_ids, self.tokenizer.sp2_token_id)

        input_idsA = input_ids.clone()
        input_idsB = input_ids.clone()
        speaker_idsA = speaker_ids.clone()
        speaker_idsB = speaker_ids.clone()

        input_idsA[speakerB] = self.emp_token_id
        input_idsB[speakerA] = self.emp_token_id
        speaker_idsA[speakerB] = 0
        speaker_idsB[speakerA] = 0

        return {
            "speakerA": {
                "input_ids": input_idsA,
                "speaker_ids": speaker_idsA,
                "token_type_ids": speaker_idsA,
                "other_token_type_ids": torch.zeros_like(speaker_idsA),
                "timings": torch.zeros_like(speaker_idsA),
            },
            "speakerB": {
                "input_ids": input_idsB,
                "speaker_ids": speaker_idsB,
                "token_type_ids": speaker_idsB,
                "other_token_type_ids": torch.zeros_like(speaker_idsB),
                "timings": torch.zeros_like(speaker_idsB),
            },
        }

    def filter_empty_turns(self, dialog):
        return [turn for turn in dialog if len(turn) > 0 or re.search(r"\w", turn)]

    def decode(self, encoded):
        return self.tokenizer.decode(encoded)

    def __call__(self, dataset):
        return self.process(dataset)


class ConversationalDM(DialogDMInterface):
    def __init__(
        self,
        tokenizer=None,
        device="cuda:0",
        save_path=None,
        load_from_cache=False,
        max_length=256,
        keep_length=64,
        overlap_length=10,
        dev_mode=False,
        parse_dialogs=None,
        combine_speaker=False,
        datasets=[Datasets.CURIOSITY_DIALOG],
        *args,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.logger.warning("No tokenizer provided. Using default tokenizer.")
            self.tokenizer = SpokenDialogTokenizer(
                tokens=["<ebc>", "<eot>", "<eint>", "<yield>", "<emp>"]
            )

        self.device = device
        self.save_path = save_path

        if self.save_path is None:
            tokenizer_string = self.tokenizer.__str__()
            if "(" in tokenizer_string:
                dirname = tokenizer_string[: tokenizer_string.index("(")]
            else:
                dirname = (
                    "GPT2TokenizerFast"
                    if "gpt" in self.tokenizer.__str__().lower()
                    else "LlamaTokenizerFast"
                )
            self.save_path = os.path.join(CACHE_PATH, "conversational", dirname)
            self.logger.warning(
                f"No save path provided. Using default save path: {self.save_path}"
            )

        self.load_from_cache_file = load_from_cache
        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length
        self.dev_mode = dev_mode
        self.parse_dialogs = parse_dialogs

        self.combine_speaker = combine_speaker

        self.processor = ConversationalProcessor(
            self.tokenizer,
            self.max_length,
            self.keep_length,
            self.overlap_length,
            self.combine_speaker,
        )

        self.data = []

        self.dataset_keys = datasets
        self.datasets = []
        self.load_datasets()

    def __str__(self):
        datasets = "-".join([Datasets(x).name for x in self.dataset_keys])
        return f"ConversationalDM_{datasets}_maxlen={self.max_length}_keep={self.keep_length}_overlap={self.overlap_length}_combine={self.combine_speaker}"

    def decode(self, encoded):
        if not isinstance(encoded, dict):
            raise ValueError("Encoded input must be a dictionary")

        if "speakerA" not in encoded:
            raise ValueError("Encoded input must contain speakerA")

        return self.tokenizer.decode(encoded["speakerA"]["input_ids"])

    def load_datasets(self):
        self.logger.info(f"Loading ConversationDM datasets {self.dataset_keys}")
        for ds in self.dataset_keys:
            if ds == Datasets.CURIOSITY_DIALOG:
                self.datasets.append(CuriosityDialog())
            elif ds == Datasets.DAILY_DIALOG:
                self.datasets.append(DailyDialog())
            elif ds == Datasets.MULTIWOZ:
                self.datasets.append(MultiWozV2())
            elif ds == Datasets.METAWOZ:
                self.datasets.append(MetaWoz())
            elif ds == Datasets.TASKMASTER1:
                self.datasets.append(Taskmaster1())
            elif ds == Datasets.TASKMASTER2:
                self.datasets.append(Taskmaster2())
            elif ds == Datasets.TASKMASTER3:
                self.datasets.append(Taskmaster3())

    def prepare_data(self):
        if self.load_from_cache_file:
            self.logger.info(
                f"Loading data from cache at file {self.get_save_load_path()}"
            )
            if self.setup():
                return

            self.logger.info("Loading data from cache failed so generate data")

        for ds in self.datasets:
            ds()

        self.dataset = ConcatDataset(self.datasets)
        self.data = self.processor.process(self.dataset)

        self.log_info()
        self.save_to_disk()

    def log_info(self):
        self.logger.info(f"Number of dialogs: {len(self.data)}")
        self.logger.info(f"Number of turns: {sum([len(d) for d in self.data])}")

    def save_to_disk(self):
        if self.save_path is None:
            return

        if not os.path.exists(self.save_path):
            self.logger.info(f"Creating directory {self.save_path}")
            os.makedirs(self.save_path)

        filename = self.get_save_load_path()
        torch.save(self, filename)
        self.logger.info(f"Saved data to {filename}")

    def setup(self):
        filename = self.get_save_load_path()
        if not os.path.exists(filename):
            return False

        saved_ds = torch.load(filename)
        self.dataset = saved_ds.dataset
        self.data = saved_ds.data
        self.dataset_keys = self.dataset_keys
        self.load_datasets()

        return True

    def reset(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise ValueError("No data found in dataset")

        return self.data[idx]

    def get_save_load_path(self):
        save_load_dir = get_abs_path(self.save_path)

        if not os.path.exists(save_load_dir):
            os.makedirs(save_load_dir)

        return os.path.join(save_load_dir, f"{self.__str__()}.pt")


if __name__ == "__main__":
    dm = ConversationalDM()
    dm.prepare_data()

    dm_idx = 0
    while True:
        enter = input("Enter to continue...")
        if enter == "q":
            break

        out = dm.decode(dm[dm_idx])
        print(out)

    print("DONE")
